from pyspark import SparkConf
from pyspark.sql import SparkSession, udf
import pyspark.sql.types as t
import pyspark.sql.functions as f

# As per instructions, sessionizing the dataset based on time (15 min)
session_window_minutes = 15


@f.udf(t.StringType())
def getSessionId(currentTimestamp, ip):
    ###
    # Creating a helper UDF, but in reality I would install PyArrow and use Pandas UDF which are vectorized in nature
    # and it is very fast compared to regular UDF
    ###

    start_block = end_block = ""
    if 0 <= currentTimestamp.minute <= 14 or (currentTimestamp.minute == 15 and currentTimestamp.second == 0):
        start_block = "00"
        end_block = "15"
    elif (currentTimestamp.minute == 15 and currentTimestamp.second > 0) or 16 <= currentTimestamp.minute <= 29 or (currentTimestamp.minute == 30 and currentTimestamp.second == 0):
        start_block = "16"
        end_block = "30"
    elif (currentTimestamp.minute == 30 and currentTimestamp.second > 0) or 31 <= currentTimestamp.minute <= 44 or (currentTimestamp.minute == 45 and currentTimestamp.second == 0):
        start_block = "31"
        end_block = "45"
    else:
        start_block = "46"
        end_block = "60"

    # Session Id is formed as yyyyMMdd-HH-start-endblock-ip
    return "{}{}{}-{}-{}{}-{}".format(currentTimestamp.strftime("%Y"), currentTimestamp.strftime("%m"),
                                     currentTimestamp.strftime("%d"), currentTimestamp.strftime("%H"),
                                     start_block, end_block, ip.replace(".", "-"))


# Enabling Shuffle service that can help in yarn container up/down scaling
conf = SparkConf() \
    .setAppName("weblog_solution") \
    .set("spark.shuffle.service.enabled", True) \
    .set("spark.dynamicAllocation.enabled", True)

spark = SparkSession\
    .builder\
    .config(conf=conf)\
    .enableHiveSupport()\
    .getOrCreate()

schema = "timestamp elb client_port backend_port request_processing_time backend_processing_time " \
         "response_processing_time elb_status_code backend_status_code received_bytes sent_bytes request " \
         "user_agent ssl_cipher ssl_protocol".split(" ")

# For simplicity, loading the data from local FS, in reality it is expected to be in HDFS or S3
# Also compression observed here is gz, its a non splittable type. For efficiency, we may choose LZO type
# compression which can be splittable
baseDF = spark\
    .read\
    .csv("../data/*.log", sep=" ", ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)\
    .toDF(*schema)\


baseDF = baseDF.withColumn('timestamp', baseDF['timestamp'].cast('timestamp'))
baseDF = baseDF.withColumn('ip', f.split(baseDF['client_port'], ':').getItem(0))
baseDF = baseDF.withColumn('session_id', getSessionId(baseDF['timestamp'], baseDF['ip'])).cache()


# 1. Sessionize the web log by IP.
# As I am copying my output here, I am sorting the output by total_hits
sessionDF = baseDF.select("timestamp", "ip", "session_id")
sessionStatsDF = sessionDF\
    .groupby("ip", "session_id")\
    .agg(f.count('*').alias('total_hits'))\
    .orderBy('total_hits', ascending=False).cache()

# For demonstration of logic, I am issuing show action, in reality we may persist to HDFS or S3 or Hive
sessionStatsDF.show(10, truncate=False)
# +-------------+------------------------------+----------+
# |ip           |session_id                    |total_hits|
# +-------------+------------------------------+----------+
# |52.74.219.71 |20150722-06-3145-52-74-219-71 |6016      |
# |119.81.61.166|20150722-12-1630-119-81-61-166|5600      |
# |106.186.23.95|20150722-17-0015-106-186-23-95|5114      |
# |119.81.61.166|20150722-12-0015-119-81-61-166|4940      |
# |106.51.132.54|20150722-12-0015-106-51-132-54|4399      |
# |52.74.219.71 |20150722-14-0015-52-74-219-71 |4277      |
# |52.74.219.71 |20150722-17-0015-52-74-219-71 |3831      |
# |119.81.61.166|20150722-14-0015-119-81-61-166|3763      |
# |119.81.61.166|20150722-13-3145-119-81-61-166|3440      |
# |119.81.61.166|20150721-22-3145-119-81-61-166|3428      |
# +-------------+------------------------------+----------+

# 2. Average session time
# I am assuming that the ask here is to compute average time spent by IP in a given session window
# As I am copying my output here, I am sorting the output by avg_session_time_in_min
avgSessionStatsBaseDF = baseDF.select("timestamp", "ip", "session_id")
avgSessionStatsDF = avgSessionStatsBaseDF.groupby("ip", "session_id").agg(f.max("timestamp").alias('max_ts'),
                                                                          f.min("timestamp").alias('min_ts'))
avgSessionStatsDF = avgSessionStatsDF.withColumn("duration_sec", avgSessionStatsDF['max_ts'].cast('long') -
                                                 avgSessionStatsDF['min_ts'].cast('long'))
avgSessionStatsDF = avgSessionStatsDF.groupby("ip").agg(f.count('session_id').alias('total_sessions'),
                                                        f.sum('duration_sec').alias('total_duration_sec'))
avgSessionStatsDF = avgSessionStatsDF.withColumn("avg_session_time_in_min",
                                                 (avgSessionStatsDF['total_duration_sec'] / 60) /
                                                 avgSessionStatsDF['total_sessions']) \
    .orderBy('avg_session_time_in_min', ascending=False)

# For demonstration of logic, I am issuing show action, in reality we may persist to HDFS or S3 or Hive
avgSessionStatsDF.show(10, truncate=False)
# +---------------+--------------+------------------+-----------------------+
# |ip             |total_sessions|total_duration_sec|avg_session_time_in_min|
# +---------------+--------------+------------------+-----------------------+
# |119.235.53.134 |1             |594               |9.9                    |
# |117.229.207.237|1             |556               |9.266666666666667      |
# |148.177.195.5  |1             |554               |9.233333333333333      |
# |103.17.82.118  |1             |551               |9.183333333333334      |
# |59.93.100.76   |1             |551               |9.183333333333334      |
# |59.89.31.116   |1             |549               |9.15                   |
# |101.60.239.65  |1             |548               |9.133333333333333      |
# |1.22.38.150    |1             |547               |9.116666666666667      |
# |117.222.221.14 |1             |547               |9.116666666666667      |
# |14.98.69.1     |1             |546               |9.1                    |
# +---------------+--------------+------------------+-----------------------+


# 3. Determine unique URL visits per session. To clarify, count a hit to a unique URL only once per session
# As I am copying my output here, I am sorting the output by num_unique_hits
uniqueURLBaseDF = baseDF.select("session_id", "request")
uniqueURLBaseDF = uniqueURLBaseDF.withColumn("url", f.split("request", " ").getItem(1))
uniqueURLDF = uniqueURLBaseDF.groupby('session_id').agg(f.countDistinct('url').alias('num_unique_hits'))\
    .orderBy('num_unique_hits', ascending=False)

# For demonstration of logic, I am issuing show action, in reality we may persist to HDFS or S3 or Hive
uniqueURLDF.show(10, truncate=False)
# +------------------------------+---------------+
# |session_id                    |num_unique_hits|
# +------------------------------+---------------+
# |20150722-12-1630-119-81-61-166|5496           |
# |20150722-06-3145-52-74-219-71 |5057           |
# |20150722-12-0015-119-81-61-166|4851           |
# |20150722-17-0015-106-186-23-95|4656           |
# |20150722-14-0015-119-81-61-166|3637           |
# |20150721-22-3145-119-81-61-166|3333           |
# |20150722-13-3145-119-81-61-166|3323           |
# |20150722-12-1630-52-74-219-71 |2967           |
# |20150722-14-0015-52-74-219-71 |2907           |
# |20150722-17-0015-119-81-61-166|2841           |
# +------------------------------+---------------+

# 4. Find the most engaged users, ie the IPs with the longest session times
# I am appending IP and Hash value of User agent, there by assuming each user agent within same ip
# corresponds to different user
# I am also assuming that we are trying to find most engaged users based on session times
# (not across all the sessions_ids, which would become most engaged user of a day)
# As I am copying my output here, I am sorting the output by duration_min
mostEngaugedBaseDF = baseDF.select("timestamp", "session_id", "ip", "user_agent")
mostEngaugedBaseDF = mostEngaugedBaseDF.withColumn("user", f.concat("ip", f.lit('_'), f.sha2("user_agent", 256)))
mostEngaugedDF = mostEngaugedBaseDF.groupby('user', 'session_id')\
    .agg((f.max('timestamp').cast('long') - f.min('timestamp').cast('long')) / 60)\
    .toDF("user", "session_id", "duration_min")\
    .orderBy("duration_min", ascending=False)

mostEngaugedDF.show(10, truncate=False)
# +--------------------------------------------------------------------------------+--------------------------------+------------------+
# |user                                                                            |session_id                      |duration_min      |
# +--------------------------------------------------------------------------------+--------------------------------+------------------+
# |111.119.199.22_f54af9f03ea52c6a4f3d0873010fa93778a1e387399baf0c331558235b47d37b |20150722-06-3145-111-119-199-22 |13.983333333333333|
# |117.220.186.227_3a5a319663e42275d264c0d49636fe3673c4ace35a759d89f400715744532cbd|20150722-06-3145-117-220-186-227|13.4              |
# |15.211.153.75_180050cb76309ecd4e9e895a18ed06b490500b93ab309126d91a9719e69097b7  |20150722-06-3145-15-211-153-75  |9.933333333333334 |
# |119.235.53.134_3a5a319663e42275d264c0d49636fe3673c4ace35a759d89f400715744532cbd |20150722-06-3145-119-235-53-134 |9.9               |
# |116.50.79.74_3a5a319663e42275d264c0d49636fe3673c4ace35a759d89f400715744532cbd   |20150722-06-3145-116-50-79-74   |9.65              |
# |52.74.219.71_3973e022e93220f9212c18d0d0c543ae7c309e46640da93a4a0314de999f5112   |20150722-06-3145-52-74-219-71   |9.316666666666666 |
# |106.186.23.95_3973e022e93220f9212c18d0d0c543ae7c309e46640da93a4a0314de999f5112  |20150722-06-3145-106-186-23-95  |9.316666666666666 |
# |52.74.219.71_043937ea8abaea325dbde1020b1bdd9f921dcbae7450dc9141c47a7d3473c917   |20150722-06-3145-52-74-219-71   |9.316666666666666 |
# |14.139.85.180_174f0a6b8501c4d65307a711a29772fe0161b3ef8482f95a789030aa3e35146f  |20150722-06-3145-14-139-85-180  |9.3               |
# |119.81.61.166_6563296a7f163a1c1c3d95555ed88fa755e8bdc852331d89cd3a8d4c173f81c4  |20150722-06-3145-119-81-61-166  |9.3               |
# +--------------------------------------------------------------------------------+--------------------------------+------------------+
