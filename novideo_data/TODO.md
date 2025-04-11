# No-Video Based Gr00T N1
As preliminary testing we are not using the video component. We plan to add it later on.

### Current Todo
- [ ] Egoview Error
- [ ] Figure out the Error with the End-Effector Arms (left_hands and right_hands)
- [ ] Ground Truth for the gestures why is it zero on the matplot?
- [ ] Change the dataset for the wave, highfive, sup to 10 - 50 samples
    - [ ] Redo the Parquet
- [ ] Redo the gestures: wave, highfive so that they are more in the POV view
- [ ] Get the mp4 Files
    - [ ] Parse the videos to the appriate fps and resolutions
    - [ ] Determine Steps per frame Correlation for Gr00t and the dataset and the video
- [ ] What are the gt_action points?
- [ ] Make the code that will lower the amount of interpolated points 
    - [ ] You can use the current way-points or other ones

### Completed ✓
- [x] Completed Export Baxter Gestures (1 of each gesture)
- [x] Completed Parquet Converter
- [x] Completed Reading the Paper
- [x] Reformat Parquet for LeRobot Format
- [x] Refactor the code for other files and got the CSV
    - [x] Change the following json files:
        - stats.json (use the stat_maker)
        - episode.jsonl (the length sizes)

### Future Works
- [ ] Standardize the FPS/speed of the interpolated movement of the arms 
  - Some gestures are faster than others, we should try to make them as standardized as possible to 
        avoid future complications
- [ ] Similarly, we should figure out the joint positioning prediction
    - There are instances when the first joint position -> next position gets screw up due to unforseen events. This causes a ripple 
    effect to all of the future join positions
- [ ] For video input testing: We can have a camera (intelRealSense etc) hooked up to the simulation so it "imitates" the actual robot engagement
    and we can figure out more of that stuff.

### Notes
- Modality #s might not be in the right order.
- Order of the join positions could be different for each of the gestures.
- The gestures are solely the gestures they do not go back to home position
- Leaving the RL components out of the stats.json file

----------------------------------------------------------

### Column Name
- [ ] Task title ~3d #type @name yyyy-mm-dd  
  - [ ] Sub-task or description  

### Completed Column ✓
- [x] Completed task title  