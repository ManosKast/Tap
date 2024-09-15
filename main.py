from time import sleep
import pyautogui as gui
import cv2
import numpy as np


# Takes a screenshot and converts it into cv2 image
def take_screenshot():
    screenshot = gui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return screenshot


def template_search(screenshot, template):
    # Ensure the template is also in grayscale
    if len(template.shape) == 3:  # If the template is in color
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    return loc


def get_random_coordinates(xloc, yloc, width, height):
    x = xloc + np.random.randint(0, width)
    y = yloc + np.random.randint(0, height)
    return x, y


# Generates a log-normal reaction time.
def human_reaction_time(mean, std_deviation):
    return np.random.lognormal(mean, std_deviation)


def main():
    # Load the template image
    template = cv2.imread('Pictures/help_img.png', cv2.IMREAD_UNCHANGED)

    # Check if the template was loaded successfully
    if template is None:
        print("Error: Template image not found. Please check the file path.")
        return

    # Convert template to grayscale if it's in color
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    while True:
        # If script is terminated, break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Main loop
        screenshot = take_screenshot()
        yloc, xloc = template_search(screenshot, template)

        # If the template is found, group the coordinates and click on it
        if len(xloc) > 0:
            rectangles, width, height = [], template.shape[1], template.shape[0]

            # Append twice, since grouping requires 2+ rectangles
            for (x, y) in zip(xloc, yloc):
                rectangles.append([int(x), int(y), width, height])
                rectangles.append([int(x), int(y), width, height])

            rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)
            # Move the mouse and click
            for (x, y, w, h) in rectangles:
                x, y = get_random_coordinates(x, y, width, height)
                sleep(human_reaction_time(np.log(0.3), 0.2))
                gui.click(x=x, y=y)

        sleep(0.5)


if __name__ == '__main__':
    main()
