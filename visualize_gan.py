import tensorflow as tf
import numpy as np
import cv2

from models import build_generator, build_deconv_gen

img_height = 32
img_width = 32
channels = 3
img_size = img_height * img_width * channels

output_path = "mnist32"

h1_size = 150
h2_size = 300
z_size = 100


def show_result(batch_res, grid_size=(8, 8), grid_pad=5, scale=4):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width, channels)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w, channels), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    img = cv2.resize(img_grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Numbers", img)


def show_img(flat_arr, scale=9):
    img = flat_arr.reshape(img_height, img_width, channels)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Numbers", img)


def test():
    z_prior = tf.placeholder(tf.float32, [1, z_size], name="z_prior")
    x_generated, _ = build_deconv_gen(z_prior, z_size)
    chkpt_fname = tf.train.latest_checkpoint(output_path)

    print chkpt_fname
    init = tf.global_variables_initializer()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, chkpt_fname)

    def randomSeed():
        return np.random.normal(0, 1, size=(1, z_size)).astype(np.float32)

    points = 100
    z_points = [randomSeed() for _ in range(points)]
    frames = 100
    i = 0
    start = 0
    end = 1
    while True:
        z_delta = (z_points[end] - z_points[start]) / frames
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_points[start] + z_delta * i})
        show_img(x_gen_val)
        key = cv2.waitKey(100)
        if key == ord('a'):
            i -= 1
            print (i)
        elif key == ord('d'):
            i += 1
            print (i)
        elif key == ord('q'):
            break
        if i >= frames:
            start += 1
            end += 1
        elif i < 0:
            start -= 1
            end -= 1
        i %= frames
        start %= points
        end %= points

if __name__ == '__main__':
    test()
