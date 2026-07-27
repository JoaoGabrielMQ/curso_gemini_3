import base64


def encode_image(image):
    with open(image, "rb") as image_file:
        encode_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encode_string

