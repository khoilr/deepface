import concurrent.futures
import cv2
import json
import multiprocessing

url = ""

manager = multiprocessing.Manager()
work_cam = manager.list()
not_work_cam = manager.list()

MAX_ATTEMPTS = 1000


def try_open_camera(url):
    for _ in range(MAX_ATTEMPTS):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            cap.release()
            return True
    return False


def run_camera(params):
    channel, subtype = params
    params = {"channel": channel, "subtype": subtype}
    param_string = f"channel={channel}&subtype={subtype}"
    full_url = f"{url}?{param_string}"

    if try_open_camera(full_url):
        work_cam.append(params)
        print(f"Camera {params} is working")
    else:
        not_work_cam.append(params)
        print(f"Camera {params} is not working")


def main():
    camera_params = [(channel, subtype) for channel in range(1, 1001) for subtype in range(0, 1001)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(run_camera, camera_params)

    # Convert the manager lists to regular lists before printing and saving
    work_cam_list = list(work_cam)
    not_work_cam_list = list(not_work_cam)

    print(f"Working cameras: {work_cam_list}")
    print(f"Not working cameras: {not_work_cam_list}")

    with open("work_cam.json", "w") as file:
        json.dump(work_cam_list, file, indent=4)

    with open("not_work_cam.json", "w") as file:
        json.dump(not_work_cam_list, file, indent=4)


if __name__ == "__main__":
    main()
