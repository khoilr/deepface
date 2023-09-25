import pathlib

from django.http import FileResponse
from django.shortcuts import render

# Create your views here.


def getImage(request, slug):
    print(slug)
    cam_dir_parent = pathlib.Path.cwd()

    image_path = cam_dir_parent.joinpath("images/frames/" + slug)
    print(image_path)
    img = open(image_path, "rb")
    respond = FileResponse(img)

    return respond


def test_camera(request):
    return render(request, "home.html")
