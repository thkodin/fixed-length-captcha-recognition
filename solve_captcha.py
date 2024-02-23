from pathlib import Path

import cv2
import numpy as np
from keras.models import load_model


def drawText(
    img: type[np.ndarray],
    text: str,
    img_bottom_left_origin: bool = False,
    text_position: int | tuple[int, int] | str = "top_left",
    text_color: tuple[int, int, int] = (0, 0, 255),
    text_font: int = cv2.FONT_HERSHEY_PLAIN,
    text_font_scale: float = 1.0,
    text_font_thickness: int = 1,
    text_line_type: int = cv2.LINE_AA,
    add_textbg: bool = True,
    textbg_symmetric_baseline_adjustment: bool = True,
    textbg_in_text_bounds: bool = False,
    textbg_color: tuple[int, int, int] = (0, 0, 0),
    textbg_thickness: int = cv2.FILLED,
    textbg_line_type: int = cv2.LINE_AA,
    textbg_padding: (
        int
        | float
        | tuple[int, int]
        | tuple[float, float]
        | tuple[int, int, int, int]
        | tuple[float, float, float, float]
    ) = 0,
):
    """
    Draw text on an image.

    Parameters
    ----------
    img : type[np.ndarray]
        The image to draw the text on.
    text : str
        The text to draw.
    img_bottom_left_origin : bool, optional
        Whether to assume image has a bottom-left origin, by default False. If True, treats the `text_position` as
        relative to bottom-left instead of the usual top-left. Therefore, `text_position="top_left"` is equivalent to
        `text_position="bottom_left"`. Only kept as a working placeholder for `bottomLeftOrigin` in OpenCV's `putText()`
        function.
    text_position : int | tuple[int, int] | str, optional
        The bottom-left corner coordinates for the text box, by default "top_left". If an int, x = y = text_pos. If a
        tuple, it should be (x, y) where x and y are the left and bottom coordinates respectively. If a string, it
        should be one of: "top_left", "top_right", "bottom_left", "bottom_right", or "center". If the text box goes out
        of image bounds, its position (and nothing else) will be adjusted within the image bounds if possible, otherwise
        an error will be thrown.
    text_color : tuple[int, int, int], optional
        The color of the text in BGR order, by default (0, 0, 255), i.e., red.
    text_font : int, optional
        The font to use, by default cv2.FONT_HERSHEY_PLAIN (integer code 1).
    text_font_scale : float, optional
        The font scale factor that is multiplied by the font-specific base size, by default 1.
    text_font_thickness : int, optional
        The thickness of the text in pixels, by default 1.
    text_line_type : int, optional
        The line type, by default cv2.LINE_AA (integer code 16).
    add_textbg : bool, optional
        Whether to add a background to the text, by default True.
    textbg_symmetric_baseline_adjustment : bool, optional
        Whether to adjust the text background box symmetrically by the baseline amount in the corresponding dimension,
        by default True. If True, when adjusting the original text box for baseline amounts in the horizontal and
        vertical dimensions, the text background box will be adjusted symmetrically by the baseline amount in both
        directions (i.e., left + right horizontally, and top + bottom vertically), otherwise the text background box
        will be adjusted non-symmetrically (i.e., only along the left horizontally, and bottom vertically) by the
        baseline amount. The baseline amount is the amount by which the bottom edge of the text box is pushed down to
        ensure descending characters like 'g' and 'y' are visible. This is only relevant if `add_textbg` is True.
    textbg_in_text_bounds : bool, optional
        Whether to adjust the text background box position to fit within the image bounds, by default False. If True,
        the text background box will be used instead of the text box to check for out-of-bounds conditions, and the
        position of the text box will be adjusted to fit within the image bounds if possible, otherwise an error will be
        thrown. Only relevant if `add_textbg` is True.
    textbg_color : tuple[int, int, int], optional
        The color of the text background in BGR order, by default (0, 0, 0), i.e., black. Only relevant if `add_textbg`
        is True.
    textbg_padding : int | float | tuple[int, int] | tuple[float, float] | tuple[int, int, int, int] | tuple[float, float, float, float], optional
        The padding of the text background, by default 0. Only relevant if `add_textbg` is True. Based on the type of
        the passed value, the padding will be interpreted as follows:

        - If an int, the padding is equal on all sides in pixels.
        - If a float, the padding is proportional to the minimum of text box dimensions.
        - If a 2-tuple, the first element is the equal horizontal padding in both left/right directions, and the second
          element is equal vertical padding in both top/bottom directions. If both elements are ints, the padding is in
          pixels, else if they are floats, the padding in each dimension is propotional to the text box size in that
          dimension.
        - If a 4-tuple, the first element is the left edge padding, the second is the top edge padding, the third is the
          right edge padding, and the fourth is the bottom edge padding. If the elements are ints, the padding is in
          pixels, else if they are floats, the padding is proportional to the corresponding text box edge.

    Sources
    -------
    - https://stackoverflow.com/a/65146731
    - https://stackoverflow.com/a/73056754
    """
    # Get image size.
    img_height, img_width = img.shape[:2]

    # Get the size of the text box given the text drawing settings. Also get the y-baseline, so we can ensure descending
    # characters ike 'g' and 'y' are visible.

    text_size, bottom_baseline = cv2.getTextSize(text, text_font, text_font_scale, text_font_thickness)
    left_baseline = 0  # this doesn't exist for typical strings, only added for sake of completeness

    # Get text box size.
    text_width, text_height = text_size

    # Figure out the position for the bottom-left corner of the text box. Anything related to the text box position we
    # want to get with respect to the text size we got from the cv2.getTextSize() function without any modification. All
    # later adjustments assume this is true.

    if isinstance(text_position, int):
        # Set the bottom-left corner to the same coordinate in both dimensions
        text_left = text_position
        text_bottom = text_position

    elif isinstance(text_position, tuple):
        # Unpack position tuple asuuming order (x, y) to get the bottom-left corner coordinates of the text box.
        text_left, text_bottom = text_position

    elif isinstance(text_position, str):
        # Set initial text positions. We'll fit them in the respective position later.
        if text_position == "top_left":
            # Text box bottom left coordinates so that left and top edges align with corresponding edges of image, need
            # to push it "down".
            text_left, text_bottom = (0, text_height)
        elif text_position == "top_right":
            # Text box bottom left coordinates so that right and top edges align with the corresponding edges of image,
            # need to pull it "left" AND push it "down".
            text_left, text_bottom = (img_width - text_width, text_height)
        elif text_position == "bottom_left":
            # Text box bottom left coordinates so that left and bottom edges align with the image's, no pull/push needed.
            text_left, text_bottom = (0, img_height)
        elif text_position == "bottom_right":
            # Text box bottom left coordinates so that right and bottom edges align with the corresponding edges of
            # image, need to pull it "left".
            text_left, text_bottom = (img_width - text_width, img_height)
        elif text_position == "center":
            # Text box bottom left coordinates so so that its center aligns with image center, initially put it to image
            # center, then pull it "left" AND push it "down" by half the text box sizes.
            text_left, text_bottom = ((img_width - text_width) // 2, (img_height + text_height) // 2)
        else:
            raise ValueError(
                "Expected `text_position` keyword argument to be one of: 'top_left', 'top_right', 'bottom_left', "
                "'bottom_right', or 'center'. Got {} instead.".format(text_position)
            )

    else:
        raise TypeError(
            "Expected `text_position` keyword argument to be an int, tuple, or string. Got {} instead.".format(
                type(text_position)
            )
        )

    if img_bottom_left_origin:
        text_bottom = img_height - text_bottom  # get the text bottom relative to the bottom edge of the image

    # Create a new text box, this one with the baseline adjustments so that characters exceeding baselines are not
    # cutoff, e.g., in vertical dimension, characters like 'g' and 'y'. Typically, the horizontal dimension has no such
    # issue, but we'll adjust it as well for the sake of completeness assuming left_baseline is 0.

    text_left_baseline_adjusted = text_left - left_baseline  # pull left edge further left
    text_bottom_baseline_adjusted = text_bottom + bottom_baseline  # push bottom edge down

    # Update height and width by double the baseline amount, so that both top and bottom, and left and right edges are
    # expanded by the same corresponding baseline amount (symmetric edge expansion). OR, just update the height and
    # width by the baseline amount, expanding only the left and bottom edges with the corresponding baseline amount
    # (non-symmetric edge expansion). To see how, consider the baseline adjusted bottom edge to be y_b' = y_b +
    # y_baseline, the original being y_b. Similarly, consider that the height of the baseline adjusted text box is h' =
    # h + y_baseline, the original being h. Then, y_top = y_b - h == y_top' = y_b' - h'. In this formulation, the top
    # edge doesn't move! However, if we were to double the baseline amount when computing h' = h + 2 * y_baseline, then
    # y_top = y_b - h != y_top' = y_b' - h' = y_b - 2 * y_baseline, therefore moving the top edge by the same amount as
    # the bottom edge. Similar logic can be applied to the left baseline (if it exists).
    if textbg_symmetric_baseline_adjustment:
        text_width_baseline_adjusted = text_width + 2 * left_baseline
        text_height_baseline_adjusted = text_height + 2 * bottom_baseline
    else:
        text_width_baseline_adjusted = text_width + left_baseline
        text_height_baseline_adjusted = text_height + bottom_baseline

    if add_textbg:
        try:
            if any(x < 0 for x in textbg_padding):
                raise ValueError("Padding must be non-negative.")
        except TypeError:
            if textbg_padding < 0:
                raise ValueError("Padding must be non-negative.")

        # What sort of padding did the user provide? Whatever we had, we need to convert it to a 4-tuple appropriately.
        if isinstance(textbg_padding, int):
            # Pad equally on all sides by this many pixels.
            textbg_pad_ltrb = (textbg_padding, textbg_padding, textbg_padding, textbg_padding)

        elif isinstance(textbg_padding, float):
            # Pad equally on all sides propotionally to the minimum of the text box width and height.
            textbg_padding = int(textbg_padding * min(text_width_baseline_adjusted, text_height_baseline_adjusted))
            textbg_pad_ltrb = (textbg_padding, textbg_padding, textbg_padding, textbg_padding)

        elif isinstance(textbg_padding, tuple):

            # We have different pad amounts.
            if len(textbg_padding) == 2:

                # Pad amounts are separate for horizontal and vertical dimensions.
                if all(isinstance(x, float) for x in textbg_padding):
                    # Pad amounts are proportional to the text box size along each dimension.
                    textbg_pad_w = int(textbg_padding[0] * text_width_baseline_adjusted)
                    textbg_pad_h = int(textbg_padding[1] * text_height_baseline_adjusted)

                elif all(isinstance(x, int) for x in textbg_padding):
                    # Pad amounts are exact (pixel units) along each dimension.
                    textbg_pad_w = textbg_padding[0]
                    textbg_pad_h = textbg_padding[1]

                else:
                    raise ValueError(
                        "Expected background padding to be either all ints or all floats. Got a mix of {} instead."
                        .format(", ".join([str(type(x)) for x in textbg_padding]))
                    )

                textbg_pad_ltrb = (textbg_pad_w, textbg_pad_h, textbg_pad_w, textbg_pad_h)

            elif len(textbg_padding) == 4:

                # Pad amounts are separate for each side.
                if all(isinstance(x, float) for x in textbg_padding):
                    # Pad amounts are proportional to the sides.
                    textbg_pad_left = int(textbg_padding[0] * text_width_baseline_adjusted)
                    textbg_pad_top = int(textbg_padding[1] * text_height_baseline_adjusted)
                    textbg_pad_right = int(textbg_padding[2] * text_width_baseline_adjusted)
                    textbg_pad_bottom = int(textbg_padding[3] * text_height_baseline_adjusted)

                elif all(isinstance(x, int) for x in textbg_padding):
                    # Pad amounts are exact (pixel units) along the sides.
                    textbg_pad_left = textbg_padding[0]
                    textbg_pad_top = textbg_padding[1]
                    textbg_pad_right = textbg_padding[2]
                    textbg_pad_bottom = textbg_padding[3]

                else:
                    raise ValueError(
                        "Expected background padding to be either all ints or all floats. Got a mix of {} instead."
                        .format(", ".join([str(type(x)) for x in textbg_padding]))
                    )

                textbg_pad_ltrb = (textbg_pad_left, textbg_pad_top, textbg_pad_right, textbg_pad_bottom)

            else:
                raise ValueError(
                    "Tuples are supported for background padding, but must be either a 2-tuple or 4-tuple. Got {}-tuple"
                    " instead.".format(len(textbg_padding))
                )

        else:
            raise TypeError(
                "Expected background padding to be an int, float, or tuple of ints and floats. Got {} instead.".format(
                    str(type(textbg_padding))
                )
            )

        # Add the background box.
        textbg_left = text_left_baseline_adjusted - textbg_pad_ltrb[0]
        textbg_bottom = text_bottom_baseline_adjusted + textbg_pad_ltrb[1]
        textbg_width = text_width_baseline_adjusted + textbg_pad_ltrb[0] + textbg_pad_ltrb[2]
        textbg_height = text_height_baseline_adjusted + textbg_pad_ltrb[1] + textbg_pad_ltrb[3]

        if textbg_in_text_bounds:
            # Update the baseline-adjusted textbox to reflect the padding.
            text_left_baseline_adjusted = textbg_left
            text_bottom_baseline_adjusted = textbg_bottom
            text_width_baseline_adjusted = textbg_width
            text_height_baseline_adjusted = textbg_height

    # First check if text box fits in the image bounds. If not, raise an error since we cannot fix this without
    # modifying the text content. We only want to perform positional adjustments if ncecessary.
    if text_width_baseline_adjusted > img_width or text_height_baseline_adjusted > img_height:
        err_msg = (
            "Text box cannot be drawn on the image as baseline-adjustments exceed the image bounds:\n"
            if not textbg_in_text_bounds
            else "Text background box cannot be drawn on the image as it exceeds image bounds:\n"
        )
        err_msg += (
            ""
            if text_width_baseline_adjusted < img_width
            else f"  - Width ({text_width_baseline_adjusted}) exceeds image width ({img_width}).\n"
        )
        err_msg += (
            ""
            if text_height_baseline_adjusted < img_height
            else f"  - Height ({text_height_baseline_adjusted}) exceeds image height ({img_height}).\n"
        )
        err_msg += (
            "Please adjust the text box contents (e.g., font, font size, thickness, etc.) to fit it.\n"
            if not textbg_in_text_bounds
            else (
                "Please consider:\n  o Not using the background in bound calculation by setting `textbg_in_text_bounds`"
                " to False\n  o Adjust text box contents (e.g., font, font size, thickness, etc.) to fit it\n  o"
                " Adjust the padding (if applied)\n"
            )
        )
        raise ValueError(err_msg)

    # Fixing the out-of-bound text cases. We will consider the baseline-adjusted text box to make the check, and update
    # the actual text box if needed.
    dx = 0
    dy = 0

    if text_left_baseline_adjusted < 0:
        # Left edge of baseline-adjusted text box is to the left of image's left edge. Automatically incorporates the
        # subset case where both the left AND right edges of the text box are to the left of the image's left edge.
        dx = text_left_baseline_adjusted
    elif text_left_baseline_adjusted + text_width_baseline_adjusted > img_width:
        # Right edge of baseline-adjusted text box is to the right of image's right edge. Automatically incorporates the
        # subset case where both the left AND right edges of the text box are to the right of the image's right edge.
        dx = text_left_baseline_adjusted + text_width_baseline_adjusted - img_width

    if text_bottom_baseline_adjusted > img_height:
        # Bottom edge of baseline-adjusted text box is below the image's bottom edge. Automatically incorporates the
        # subset case where both the top AND bottom edges of the text box are below the image's bottom edge.
        dy = text_bottom_baseline_adjusted - img_height
    elif text_bottom_baseline_adjusted - text_height_baseline_adjusted < 0:
        # Top edge of baseline-adjusted text box is above image's top edge. Automatically incorporates the subset
        # case where both the top AND bottom edges of the text box are above the image's top edge.
        dy = text_bottom_baseline_adjusted - text_height_baseline_adjusted

    # Align the baseline-adjusted text box edges with the image edges.
    text_left_baseline_adjusted -= dx
    text_bottom_baseline_adjusted -= dy

    # Update the original text box's edges to reflect the change.
    text_left -= dx
    text_bottom -= dy

    if add_textbg:
        # Update the background box's edges to reflect the change.
        textbg_left -= dx
        textbg_bottom -= dy

        textbg_topleft = (textbg_left, textbg_bottom - textbg_height)
        textbg_bottomright = (textbg_left + textbg_width, textbg_bottom)

        # Draw a filled rectangle to serve as the text background.
        cv2.rectangle(
            img,
            textbg_topleft,
            textbg_bottomright,
            textbg_color,
            thickness=textbg_thickness,
            lineType=textbg_line_type,
        )

    # Draw the text on the image.
    cv2.putText(
        img,
        text,
        (text_left, text_bottom),
        text_font,
        text_font_scale,
        text_color,
        thickness=text_font_thickness,
        lineType=text_line_type,
    )

    return img


if __name__ == "__main__":

    # Configurable.
    MODEL_NAME = "BEST-CaptchaSolver-epoch_70-vloss_3.44.h5"

    # Directories.
    DIR_MODELS = Path("models")
    DIR_CAPTCHA_IMAGES = Path("captcha-dataset/inference")
    DIR_RESULTS = Path("preds")
    PATH_MODEL = DIR_MODELS / MODEL_NAME

    # All lowercase characters + digits that are used in the CAPTCHA images
    CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"

    if not PATH_MODEL.exists():
        raise FileNotFoundError(f"Model file '{MODEL_NAME}' does not exist in the directory '{PATH_MODEL.parent}'.")

    if not DIR_CAPTCHA_IMAGES.exists():
        raise FileNotFoundError(f"CAPTCHA image directory '{DIR_CAPTCHA_IMAGES}' does not exist.")

    # Get the paths to the CAPTCHA images.
    captcha_images = list(DIR_CAPTCHA_IMAGES.glob("*.png"))

    # Load the trained CAPTCHA solver model.
    model = load_model(PATH_MODEL)

    # Get the input shape of the model.
    input_shape = model.layers[0].input_shape[0]  # batch (0), height (1), width (2), channels (3)
    img_height, img_width = input_shape[1:3]

    # Open a TextIO and move the cursor to the start of the file.
    resultfile = open(DIR_RESULTS / "_preds.csv", "w")
    resultfile.truncate(0)

    # Load the CAPTCHA image to solve, and preprocess it the same way as at train time.
    for img_path in captcha_images:

        img = cv2.imread(str(img_path))

        height, width, _ = img.shape

        # Grayscale and normalize.
        img_input = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        img_input = img_input / 255.0

        # Resize to model height, width, and add the batch (first) and channel (last) dimensions.
        img_input = cv2.resize(img_input, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        img_input = img_input[np.newaxis, ..., np.newaxis]

        # Predict the CAPTCHA text. This is returned as a (batch, captcha_len, num_chars) shaped array.
        prediction = model.predict(img_input)

        # Now we want to reconstruct the text from the prediction. The prediction is (captcha_len, num_chars), so
        # prediction[i, j] refers to the probability of the j-th character being at position i in the CAPTCHA. We want
        # to grab the most probable among all j for each i.
        predicted_text = ""
        for i in range(len(prediction)):
            predicted_text += CHARSET[np.argmax(prediction[i])]

        print(f"Predicted CAPTCHA text: {predicted_text} | True: {img_path.stem}")

        resultfile.write(f"{img_path},{predicted_text}\n")

        # Save an image with the predicted text drawn on the top-right corner.
        img = drawText(
            img,
            predicted_text,
            img_bottom_left_origin=False,
            text_position="bottom_right",  # the function will auto-adjust within image bounds
            text_color=(0, 255, 255),
            text_font=cv2.FONT_HERSHEY_PLAIN,
            text_font_scale=1.0,
            text_font_thickness=1,
            text_line_type=cv2.LINE_AA,
            add_textbg=True,
            textbg_in_text_bounds=True,
            textbg_color=(255, 0, 0),
            textbg_thickness=cv2.FILLED,
            textbg_line_type=cv2.LINE_AA,
            textbg_padding=0,
        )

        cv2.imwrite(str(DIR_RESULTS / f"{img_path.stem}_pred.png"), img)

    print("Results saved to directory:", DIR_RESULTS)

    resultfile.close()
