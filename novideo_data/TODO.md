# No-Video Based Gr00T N1
As preliminary testing we are not using the video component. We plan to add it later on.

### Current Todo
- [ ] Reformat Parquet for LeRobot Format
- [ ] Change the following json files:
    - stats.json (use the stat_maker)
    - episode.jsonl (the length sizes)
- [ ] Load the dataset on the Gr00T

### Completed Column ✓
- [✓] Completed Export Baxter Gestures (1 of each gesture)
- [✓] Completed Parquet Converter
- [✓] Completed Reading the Paper

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



### Column Name
- [ ] Task title ~3d #type @name yyyy-mm-dd  
  - [ ] Sub-task or description  

### Completed Column ✓
- [x] Completed task title  