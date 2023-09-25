from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse 
from dotenv import load_dotenv
import os
from .models import VideoCamera
load_dotenv()
URL = os.getenv("URL")

from django.conf import settings
import os 


def get_images(file: int | str) -> bytearray:
    """
    get image from media folder.

    Args:
        file (int | str): File name.

    Returns:
        bytearray: File bytes.
    """
    base_dir = settings.MEDIA_ROOT    
    my_file = os.path.join(base_dir, f"{file}.png")
    with open(my_file, "rb") as image:
        f = image.read()
        b = bytearray(f)
        return b  

def gen():
    """
    Generate image collection streamming.

    Yields:
        _type_: _description_
    """
    while True:
        for i in range(1, 21):
            frame = get_images(i)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def mask_feed(request):
    """
    Image datastream to request.

    Args:
        request (request): request session.

    Returns:
        StreamingHttpResponse: image datastream.
    """
    return StreamingHttpResponse(gen(),
                    content_type='multipart/x-mixed-replace; boundary=frame')

def test(request) -> HttpResponse:
    """
    Render home.html for user.

    Args:
        request (request): request session.

    Returns:
        HttpResponse: HttpResponse.
    """
    return render(request, "home.html")

def gen_video():
    """
    Generate image collection streamming.

    Yields:
        _type_: _description_
    """
    camera = VideoCamera(URL=URL)
    while True:
        try:
            frame = camera.get_frame()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except:
            camera = VideoCamera(URL=URL)
        
    
        
def mask_feed_video(request):
    """
    Image datastream to request.

    Args:
        request (request): request session.

    Returns:
        StreamingHttpResponse: image datastream.
    """
    return StreamingHttpResponse(gen_video(),
                    content_type='multipart/x-mixed-replace; boundary=frame')

def test_video(request) -> HttpResponse:
    """
    Render video.html for user.

    Args:
        request (request): request session.

    Returns:
        HttpResponse: HttpResponse.
    """
    return render(request, "video.html")
