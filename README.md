# traffic_vid_annotations

Workflow tutorial:<br>

1.) Click on the vehicles in frame to select for segmentation <br>
2.) Enter their id's in the prompt<br>
3.) Add more positive or negative selections of your vehicles until the segmentation masks are satisfactory<br>
4.) Click Enter to go to the next frame<br>
5.) starting from second frame you should see the second window that displays bounding boxes<br>
6.) when you'd like a vehicle to be cropped out of the frame, click the bounding box<br>
7.) Once a vehicle with a new id will need to be tracked, you will have to select the new vehicle and re-select all the ones that were still tracked in the frame, as the model will not start tracking new vehicles in an established tracking scene.<br>
8.) Press q to exit program<br>


Notes:<br>

1.) Lai atlasītu mašīnu segmentacijai - nospiež kreiso peles taustiņu ( parādās zaļa zvaigzne)<br>
2.) Lai pateiktu modelim, kur NAV daļa no mašīnas - nospiež labo taustiņu (parādās sarkana zvaigzne)<br>
