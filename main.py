from pipelines.video_pipeline import video_pipeline

def main():
    video_path, ad_image_path = 'video_lacy.mp4', 'images/clix.png'
    video_pipeline(video_path, ad_image_path)

if __name__ == '__main__':
    main()