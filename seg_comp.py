import cv2
import numpy as np
from miseval import evaluate
from os import listdir
from os.path import isfile, join, isdir
from os import mkdir
import re
import matplotlib.pyplot as plt
import pandas as pd


# Plot image with the option of a title. The image is displayed until any key is pressed
def plot_img(image, title=""):
    cv2.imshow(title, image)
    cv2.waitKey(0)


# Add text directly into the image
def add_text(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    return_dist = 30
    fontsize = 1
    fontolor = (255, 255, 255)
    thickness = 1
    linetype = 2
    for row in range(len(text)):
        location = (10, 50 + return_dist * row)
        cv2.putText(img, text[row], location, font, fontsize, fontolor, thickness, linetype)


# Create a sample mask until I have the data I need
def create_sample_mask(sample_img):
    sample_image = cv2.imread(sample_img)
    sample_trained_mask = np.zeros_like(sample_image)
    [x, y, _] = sample_trained_mask.shape
    lower_factor = 3.5 / 8
    upper_factor = 2.5 / 8
    [lx, ly] = [int(x * lower_factor), int(y * lower_factor)]
    cv2.rectangle(sample_trained_mask, (lx, ly), (lx + int(x * upper_factor), ly + int(y * upper_factor)), (255, 255, 255), -1)
    cv2.imwrite("sample_trained_mask.png", sample_trained_mask)

    sample_annote_mask = np.zeros_like(sample_image)
    lower_factor = 3 / 8
    upper_factor = 2 / 8
    [lx, ly] = [int(x * lower_factor), int(y * lower_factor)]
    cv2.rectangle(sample_annote_mask, (lx, ly), (lx + int(x * upper_factor), ly + int(y * upper_factor)), (255, 255, 255), -1)
    cv2.imwrite("sample_annote_mask.png", sample_annote_mask)

# create_sample_mask(r"C:\Users\Alex Devlin\PycharmProjects\segmentation_comparison\Unlabelled Imgs\05_STIM003_1_35.0_Plac_1.jpeg")


def calc_dice(im1, im2):

    # Normalize Images
    im1[im1 > 1] = 1
    im2[im2 > 1] = 1

    # im1 = cv2.normalize(im1, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # im2 = cv2.normalize(im2, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    dice = evaluate(im1, im2, metric="DSC")
    iou = evaluate(im1, im2, metric="IoU")
    pixel_area = np.sum(im1)

    return dice, iou, pixel_area


# Finds the GA given the title of a scan or mask. Assume the GA is in the format WW.D
def find_ga(image_title):
    ga_search = re.search("\d\d\.\d", image_title)
    ga = float(ga_search.group())
    ga_decimal = ga - ga % 1 + ga % 1 * 10 / 7
    return ga_decimal


# Checks the givens folders. Files from unlablled_folder are used for naming. Files of the same name are checked in
# the other folders. A composite image showing overlap is saved in comparison folder along with DICE score
def compare_masks(unlabelled_folder, annote_folder, trained_folder, comparison_folder):

    dice_values = []
    ga_values = []
    area_values = []

    unlabelled_imgs = [f for f in listdir(unlabelled_folder) if isfile(join(unlabelled_folder, f))]
    annotated_masks = [f for f in listdir(annote_folder) if isfile(join(annote_folder, f))]
    trained_masks = [f for f in listdir(trained_folder) if isfile(join(trained_folder, f))]

    unlabelled_imgs = [f for f in unlabelled_imgs if f.endswith('.png')]
    annotated_masks = [f for f in annotated_masks if f.endswith('.png')]
    trained_masks = [f for f in trained_masks if f.endswith('.png')]

    unlabelled_imgs.sort()
    trained_masks.sort()
    annotated_masks.sort()

    # Excluded masks due to low quality, incorrect labelling, etc
    # excluded = ["106_T2_1_19.3_Liver_1_U_7.8cm"]
    try:
        for ex_file in excluded:
            trained_masks.remove(ex_file)
    except NameError:
        pass

    # Create a new folder which will contain the images comeparing the masks and some Data Metrics
    if not isdir(comparison_folder):
        mkdir(comparison_folder)

    for i, file in enumerate(trained_masks):

        # unlabelled_img = cv2.imread(join(unlabelled_folder, file))
        unlabelled_img = cv2.imread(join(unlabelled_folder, unlabelled_imgs[i]))
        # trained_mask = cv2.imread(join(trained_folder, file))
        trained_mask = cv2.imread(join(trained_folder, file))
        # annotated_mask = cv2.imread(join(annote_folder, file))
        annotated_mask = cv2.imread(join(annote_folder, annotated_masks[i]))

        ga_values.append(find_ga(file))

        # create a copy of the unannotated image which will have the mask regions as colored overlays.
        # Green common area covered by both masks
        # Red annotation mask area but not trained mask area
        # Blue trained mask area but not annotated mask area
        mask_overlap = np.copy(unlabelled_img)

        # area covtred by both masks. Highlighted in Green.
        mask_overlap[(cv2.bitwise_and(trained_mask, annotated_mask) == 255).all(-1)] = [0, 255, 0]

        # area covered by the annoated mask that was not covered by the trained mask. Highlighted in Blue
        mask_overlap[(cv2.bitwise_xor(annotated_mask, trained_mask) - annotated_mask == 255).all(-1)] = [255, 0, 0]

        # area covered by mask from the model that was not covered by the mask from annotation. Highlighted in Red
        mask_overlap[(cv2.bitwise_xor(annotated_mask, trained_mask) - trained_mask == 255).all(-1)] = [0, 0, 255]

        mask_overlap = cv2.addWeighted(mask_overlap, 0.3, unlabelled_img, 0.7, 0, mask_overlap)
        dice, iou, pixel_area = calc_dice(trained_mask, annotated_mask)
        add_text(mask_overlap, ["DICE Score: " + str(round(dice, 3))])
        dice_values.append(dice)
        area_values.append(pixel_area)

        out_image = np.concatenate([unlabelled_img, mask_overlap], axis=1)
        cv2.imwrite(join(comparison_folder, file), out_image)

    # print("DICE Mean:", round(np.mean(dice_values), 3))
    # print("DICE STD:", round(np.std(dice_values), 3))
    f = open(join(comparison_folder, "DICE Mean and STD.txt"), "w")
    f.write("DICE Mean: " + str(round(np.mean(dice_values), 3)) + "\n")
    f.write("DICE STD: " + str(round(np.std(dice_values), 3)))
    f.close()

    plt.scatter(ga_values, dice_values)
    plt.title("DICE vs Gest Age")
    plt.xlabel("Gestational Age (Weeks)")
    plt.ylabel("DICE Score")
    plt.savefig(join(comparison_folder, "DICE vs Gest Age.png"))
    plt.close()
    # plt.show()

    plt.hist(dice_values, 20)
    plt.title("DICE Score")
    plt.savefig(join(comparison_folder, "DICE Histogram.png"))
    plt.close()
    plt.show()

    df = pd.DataFrame({'Filename': trained_masks, 'Pixel Area': area_values, 'DICE Score': dice_values})
    df.to_csv(join(comparison_folder, "Metrics.csv"))



# Define folder locations of the masks and images being processed
root = r"C:\Users\Alex Devlin\Desktop\STIM Data\Chinook Sync\Sonogram Shenanigan Squad\placenta"
unlabelled_folder = join(root, "images")
annote_folder = join(root, "annotations")
trained_folder = join(root, "ensemble_placenta")
comparison_folder = join(root, "Masks Comparison")

compare_masks(unlabelled_folder, annote_folder, trained_folder, comparison_folder)

