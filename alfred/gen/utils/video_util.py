import subprocess

class VideoSaver(object):

    def __init__(self, frame_rate=5):
        self.frame_rate = frame_rate

    def save(self, image_path, save_path):
        subprocess.call(["ffmpeg -r %d -pattern_type glob -y -i '%s' -c:v libx264 -pix_fmt yuv420p '%s'" %
                         (self.frame_rate, image_path, save_path)], shell=True)