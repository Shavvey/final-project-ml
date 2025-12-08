import scripts.stats as stats
import scripts.images as imgs


def main():
    # stats.confusion_matrix_rgb_cnn()
    imgs.image_color_histogram()
    imgs.image_color_histogram_normalized()


if __name__ == "__main__":
    main()
