import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format

config = configparser.ConfigParser()
config.read_file(open('dl.cfg'))

config.sections()
os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']

def create_spark_session():
    """Creates a spark session"""
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.5") \
        .getOrCreate()
    return spark

def process_song_data(spark, input_data, output_data):
    """Reads song_data into a dataframe then into a songs_table
     table and artists_table. Writes both tables to parquet.
    """
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"

    # read song data file
    df = spark.read.json(song_data)

    # create song_data_table view for SQL
    df.createOrReplaceTempView("song_data_table")

    # extract columns to create songs table
    songs_table = spark.sql("""
                            SELECT
                                song_id,
                                title,
                                artist_id,
                                year,
                                duration
                            FROM song_data_table
                            WHERE song_id IS NOT NULL
                            """)

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode('overwrite').partitionBy("year", "artist_id").parquet(output_data +"songs_table.parquet")

    # extract columns to create artists table
    artists_table = spark.sql("""
                                SELECT DISTINCT
                                    artist_id,
                                    artist_name,
                                    artist_location,
                                    artist_latitude,
                                    artist_longitude
                                FROM song_data_table
                                WHERE artist_id IS NOT NULL
                            """)

    # write artists table to parquet files
    artists_table.write.mode('overwrite').parquet(output_data +'artists_table.parquet')

def process_log_data(spark, input_data, output_data):
    """Reads log_data into a dataframe then into a users_table
    and a time_table. Writes both tables to parquet. The time_table
    is created by extracting and populating relevant columns from origin
    timestamp column.
    Then the songplays_table fact table is populated and written into parquet.
    """
    # get filepath to log data file
    log_data = input_data + 'log_data/*.json'

    # read log data file
    df = spark.read.json(log_data)

    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # create log_data_table view for SQL
    df.createOrReplaceTempView("log_data_table")

    # extract columns for users table
    users_table = spark.sql("""
                            SELECT DISTINCT
                                (log_data_table.userId) AS user_id,
                                log_data_table.firstName AS first_name,
                                log_data_table.lastName AS last_name,
                                log_data_table.gender AS gender,
                                log_data_table.level AS level
                            FROM log_data_table
                            WHERE log_data_table.userId IS NOT NULL
                        """)

    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(output_data +'users_table.parquet')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x) / 1000)))
    df = df.withColumn("timestamp", get_timestamp(df.ts))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000.0)))
    df = df.withColumn("datetime", get_datetime(df.ts))

    # extract columns to create time table
    time_table = df.select(
            'timestamp',
            hour('datetime').alias('hour'),
            dayofmonth('datetime').alias('day'),
            weekofyear('datetime').alias('week'),
            month('datetime').alias('month'),
            year('datetime').alias('year'),
            date_format('datetime', 'F').alias('weekday')
    )

    # extract columns to create time table
    time_table = spark.sql("""
                            SELECT
                                tt.time as start_time,
                                hour(tt.time) as hour,
                                dayofmonth(tt.time) as day,
                                weekofyear(tt.time) as week,
                                month(tt.time) as month,
                                year(tt.time) as year,
                                dayofweek(tt.time) as weekday
                            FROM (
                                SELECT
                                    to_timestamp(log_data_table.ts/1000) as time
                                FROM log_data_table
                                WHERE log_data_table.ts IS NOT NULL
                            ) tt
                        """)

    # write time table to parquet files partitioned by year and month
    time_table.write.mode('overwrite').partitionBy("year", "month").parquet(output_data +'time_table.parquet')

    # read in song data to use for songplays table
    song_data = input_data + "song_data/*/*/*/*.json"
    song_df = spark.read.json(song_data)

    # create song_data_table view for SQL
    song_df.createOrReplaceTempView("song_data_table")

    # extract columns from joined song and log datasets to create songplays table
    songplays_table = spark.sql("""
                                    SELECT
                                        monotonically_increasing_id() AS songplay_id,
                                        to_timestamp(log_data_table.ts/1000) AS start_time,
                                        month(to_timestamp(log_data_table.ts/1000)) AS month,
                                        year(to_timestamp(log_data_table.ts/1000)) AS year,
                                        log_data_table.userId AS user_id,
                                        log_data_table.level AS level,
                                        song_data_table.song_id AS song_id,
                                        song_data_table.artist_id AS artist_id,
                                        log_data_table.sessionId AS session_id,
                                        log_data_table.location AS location,
                                        log_data_table.userAgent AS user_agent
                                    FROM log_data_table
                                    JOIN song_data_table
                                        ON log_data_table.artist = song_data_table.artist_name
                                        AND log_data_table.song = song_data_table.title
                                """)

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.mode('overwrite').partitionBy("year", "month").parquet(output_data +'songplays_table.parquet')

def main():
    """Main function that calls the functions above
    """

    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://dend-14/"

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
