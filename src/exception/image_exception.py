class ImageException(Exception):

    def __init__(self, exp_type: str):
        if exp_type == 'path':
            super().__init__('The file doesnt exists. please try again with a correct path.')

        elif exp_type == 'format':
            super().__init__('you must address a file with .jpg or .png extension.')

        else:  # exp_type == 'size'
            print('size of image must be 28*28 pixels')
