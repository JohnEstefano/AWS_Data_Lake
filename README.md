# AWS_Data_Lake
## Introduction
A music streaming startup, Sparkify, has grown their user base and song database and wants to move their processes and data onto the cloud. Their data resides in S3, in a directory of JSON logs on user activity on the app, as well as a directory with JSON metadata on the songs in their app.

As their data engineer, I was tasked with building an ETL pipeline that extracts their data from S3, processes them using Spark, and loads the data back into S3 as a set of dimensional tables. This will allow their analytics team to continue finding insights in what songs their users are listening to.


## Project Datasets
I worked with two datasets that reside in S3. Here are the S3 links for each:

>Song data: s3://udacity-dend/song_data

>Log data: s3://udacity-dend/log_data

### Song Dataset
The first dataset is a subset of real data from the Million Song Dataset. Each file is in JSON format and contains metadata about a song and the artist of that song. The files are partitioned by the first three letters of each song's track ID. For example, here are filepaths to two files in this dataset.

song_data/A/B/C/TRABCEI128F424C983.json
song_data/A/A/B/TRAABJL12903CDCF1A.json

### Log Dataset
The second dataset consists of log files in JSON format generated by this event simulator based on the songs in the dataset above. These simulate app activity logs from an imaginary music streaming app based on configuration settings.

The log files in the dataset I worked with are partitioned by year and month. For example, here are filepaths to two files in this dataset.

log_data/2018/11/2018-11-12-events.json
log_data/2018/11/2018-11-13-events.json

## Schema for Song Play Analysis
Using the song and event datasets, I created a star schema optimized for queries on song play analysis. It consists of two staging tables, one fact table and four dimension tables. Pyspark SQL was used to create and manipulate the data. The tables are detailed below:

### Fact Table
1. TABLE NAME: songplays
- TABLE COLUMNS: songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent


### Dimension Tables
1. TABLE NAME: users
- TABLE COLUMNS: user_id, first_name, last_name, gender, level

2. TABLE NAME: songs
- TABLE COLUMNS: song_id, title, artist_id, year, duration

3. TABLE NAME: artists  
- TABLE COLUMNS: artist_id, artist_name, artist_location, artist_latitude, artist_longitude

4. TABLE NAME: time
- TABLE COLUMNS: start_time, hour, day, week, month, year, weekday

## REPO CONTENTS

*etl.py* starts ETL that loads data from S3 into spark for processing and then back into S3 as parquet files.

*dl.cfg* contains your AWS credentials (not included, gitignore)

*README.md* project description
