In episodes:
index = #
task = description, real
length = # of csv lines (Total line - 1 :: Does not count the first line (the names))

In tasks:
id     = numeric task ID
language = natural language description of the task

In modality:
dim    = total size of the joint vector (e.g., 7, 14)
split  = named sections of the vector (e.g., left_arm, right_arm)
name   = label for that group of joints (optional but useful)

In info:
dataset_name     = name of your dataset
version          = version string (e.g. "1.0")
description      = a summary of what's in the dataset
fps              = frame rate (e.g. 20.0)
features         = data structure per field (shape, dtype, names)
data_path        = where GR00T finds parquet files
video_path       = optional, for video files (null if unused)


"data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
"video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",

chunk:03d and episodes:06d = decimal places (Example: 000, 0000000)


For the Stats stuff:
Field	Keep it if...
timestamp	            You're modeling things over time
task_index	            You're training a multi-task model
annotation.*	        You're using language/text input or filtering data
next.reward, next.done	You're using reinforcement learning
index, episode_index	Useful for tracking/debugging but not required

