import os
import json


def create_student_folder(roll):

    folder = os.path.join(
        "data",
        "students",
        roll
    )

    images = os.path.join(folder, "images")

    os.makedirs(images, exist_ok=True)

    return folder, images


def save_student_info(student, folder):

    with open(
        os.path.join(folder, "info.json"),
        "w"
    ) as file:

        json.dump(
            student.__dict__,
            file,
            indent=4
        )