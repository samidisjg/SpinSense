
def convert_pixel_distance_to_meters(pixel_distance, refrence_height_in_meters, reference_height_in_pixels):
    return (pixel_distance * refrence_height_in_meters) / reference_height_in_pixels

def convert_meters_to_pixel_distance(meters, refrence_height_in_meters, reference_height_in_pixels):
    return (meters * reference_height_in_pixels) / refrence_height_in_meters

