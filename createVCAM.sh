 sudo rmmod v4l2loopback
 sudo modprobe v4l2 loopback devices=1 video_nr=4 card_label="DLC Webcam" exclusive_caps=0 max_buffers=2
