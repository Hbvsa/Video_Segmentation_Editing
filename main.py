from pipelines.video_pipeline import video_pipeline

def main():

    video_path, ad_image_path, max_frames, seg_per_sec, show = 'jason.mp4', 'images/clix.png', 60*10, 1, True
    video_pipeline(video_path, ad_image_path, max_frames, seg_per_sec, show)

if __name__ == '__main__':
    main()