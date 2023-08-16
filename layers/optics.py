import abc

import tensorflow as tf
import numpy as np

from numpy.fft import ifftshift
import fractions
import poppy

##############################
# Helper functions
##############################

def get_zernike_volume(resolution, n_terms, scale_factor=1e-5):
    zernike_volume = poppy.zernike.zernike_basis(nterms=n_terms, npix=resolution, outside=0.0)
    return zernike_volume * scale_factor

def fspecial(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def zoom(image_batch, zoom_fraction):
    """Get central crop of batch
    """
    images = tf.unstack(image_batch, axis=0)
    crops = []
    for image in images:
        crop = tf.image.central_crop(image, zoom_fraction)
        crops.append(crop)
    return tf.stack(crops, axis=0)


def transp_fft2d(a_tensor, dtype=tf.complex64):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    # Tensorflow's fft only supports complex64 dtype
    a_tensor = tf.cast(a_tensor, tf.complex64)
    # Tensorflow's FFT operates on the two innermost (last two!) dimensions
    a_tensor_transp = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_fft2d = tf.fft2d(a_tensor_transp)
    a_fft2d = tf.cast(a_fft2d, dtype)
    a_fft2d = tf.transpose(a_fft2d, [0, 2, 3, 1])
    return a_fft2d


def transp_ifft2d(a_tensor, dtype=tf.complex64):
    a_tensor = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft2d_transp = tf.ifft2d(a_tensor)
    # Transpose back to [batch_size, x, y, channels]
    a_ifft2d = tf.transpose(a_ifft2d_transp, [0, 2, 3, 1])
    a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d


def compl_exp_tf(phase, dtype=tf.complex64, name='complex_exp'):
    """Complex exponent via euler's formula, since Cuda doesn't have a GPU kernel for that.
    Casts to *dtype*.
    """
    phase = tf.cast(phase, tf.float64)
    return tf.add(tf.cast(tf.cos(phase), dtype=dtype),
                  1.j * tf.cast(tf.sin(phase), dtype=dtype),
                  name=name)


def laplacian_filter_tf(img_batch):
    """Laplacian filter. Also considers diagonals.
    """
    laplacian_filter = tf.constant([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=tf.float32)
    laplacian_filter = tf.reshape(laplacian_filter, [3, 3, 1, 1])

    filter_input = tf.cast(img_batch, tf.float32)
    filtered_batch = tf.nn.convolution(filter_input, filter=laplacian_filter, padding="SAME")
    return filtered_batch


def laplace_l1_regularizer(scale):
    if np.allclose(scale, 0.):
        print("Scale of zero disables the laplace_l1_regularizer.")

    def laplace_l1(a_tensor):
        with tf.name_scope('laplace_l1_regularizer'):
            laplace_filtered = laplacian_filter_tf(a_tensor)
            laplace_filtered = laplace_filtered[:, 1:-1, 1:-1, :]
            attach_summaries("Laplace_filtered", tf.abs(laplace_filtered), image=True, log_image=True)
            return scale * tf.reduce_mean(tf.abs(laplace_filtered))

    return laplace_l1


def laplace_l2_regularizer(scale):
    if np.allclose(scale, 0.):
        print("Scale of zero disables the laplace_l1_regularizer.")

    def laplace_l2(a_tensor):
        with tf.name_scope('laplace_l2_regularizer'):
            laplace_filtered = laplacian_filter_tf(a_tensor)
            laplace_filtered = laplace_filtered[:, 1:-1, 1:-1, :]
            attach_summaries("Laplace_filtered", tf.abs(laplace_filtered), image=True, log_image=True)
            return scale * tf.reduce_mean(tf.square(laplace_filtered))

    return laplace_l2


def phaseshifts_from_height_map(height_map, wave_lengths, refractive_idcs):
    '''Calculates the phase shifts created by a height map with certain
    refractive index for light with specific wave length.
    '''
    # refractive index difference
    delta_N = refractive_idcs.reshape([1, 1, 1, -1]) - 1.
    # wave number
    wave_nos = 2. * np.pi / wave_lengths
    wave_nos = wave_nos.reshape([1, 1, 1, -1])
    # phase delay indiced by height field
    phi = wave_nos * delta_N * height_map
    phase_shifts = compl_exp_tf(phi)
    return phase_shifts


def get_one_phase_shift_thickness(wave_lengths, refractive_index):
    """Calculate the thickness (in meter) of a phaseshift of 2pi.
    """
    # refractive index difference
    delta_N = refractive_index - 1.
    # wave number
    wave_nos = 2. * np.pi / wave_lengths

    two_pi_thickness = (2. * np.pi) / (wave_nos * delta_N)
    return two_pi_thickness


def attach_summaries(name, var, image=False, log_image=False):
    if image:
        tf.summary.image(name, var, max_outputs=3)
    if log_image and image:
        tf.summary.image(name + '_log', tf.log(var + 1e-12), max_outputs=3)
    tf.summary.scalar(name + '_mean', tf.reduce_mean(var))
    tf.summary.scalar(name + '_max', tf.reduce_max(var))
    tf.summary.scalar(name + '_min', tf.reduce_min(var))
    tf.summary.histogram(name + '_histogram', var)


def fftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        split = (input_shape[axis] + 1) // 2
        mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def ifftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def psf2otf(input_filter, output_size):
    '''Convert 4D tensorflow filter into its FFT.

    :param input_filter: PSF. Shape (height, width, num_color_channels, num_color_channels)
    :param output_size: Size of the output OTF.
    :return: The otf.
    '''
    # pad out to output_size with zeros
    # circularly shift so center pixel is at 0,0
    fh, fw, _, _ = input_filter.shape.as_list()

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1

        padded = tf.pad(input_filter, [[pad_top, pad_bottom],
                                       [pad_left, pad_right], [0, 0], [0, 0]], "CONSTANT")
    else:
        padded = input_filter

    padded = tf.transpose(padded, [2, 0, 1, 3])
    padded = ifftshift2d_tf(padded)
    padded = tf.transpose(padded, [1, 2, 0, 3])

    ## Take FFT
    tmp = tf.transpose(padded, [2, 3, 0, 1])
    if tmp.dtype == tf.complex64:
        tmp = tf.fft2d(tmp)
    else:
        tmp = tf.fft2d(tf.complex(tmp, 0.))
    return tf.transpose(tmp, [2, 3, 0, 1])


def next_power_of_two(number):
    closest_pow = np.power(2, np.ceil(np.math.log(number, 2)))
    return closest_pow


def img_psf_conv(img, psf, otf=None, adjoint=False, circular=False):
    '''Performs a convolution of an image and a psf in frequency space.

    :param img: Image tensor.
    :param psf: PSF tensor.
    :param otf: If OTF is already computed, the otf.
    :param adjoint: Whether to perform an adjoint convolution or not.
    :param circular: Whether to perform a circular convolution or not.
    :return: Image convolved with PSF.
    '''
    img = tf.convert_to_tensor(img)
    psf = tf.convert_to_tensor(psf)

    img_shape = img.shape.as_list()

    if not circular:
        target_side_length = 2 * img_shape[1] #2res

        height_pad = (target_side_length - img_shape[1]) / 2 # 1/2 res
        width_pad = (target_side_length - img_shape[1]) / 2

        pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
        pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))

        img = tf.pad(img, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "CONSTANT")
        img_shape = img.shape.as_list() # 2res

    img_fft = transp_fft2d(img)

    if otf is None:
        otf = psf2otf(psf, output_size=img_shape[1:3])
        otf = tf.transpose(otf, [2, 0, 1, 3])

    otf = tf.cast(otf, tf.complex64)
    img_fft = tf.cast(img_fft, tf.complex64)

    if adjoint:
        result = transp_ifft2d(img_fft * tf.conj(otf))
    else:
        result = transp_ifft2d(img_fft * otf)

    #result = tf.cast(tf.real(result), tf.float32)

    if not circular:
        result = result[:, pad_top:-pad_bottom, pad_left:-pad_right, :]

    return result


def depth_dep_convolution(img, psfs, disc_depth_map):
    """Convolves an image with different psfs at different depths as determined by a discretized depth map.

    Args:
        img: image with shape (batch_size, height, width, num_img_channels)
        psfs: filters with shape (kernel_height, kernel_width, num_img_channels, num_filters)
        disc_depth_map: Discretized depth map.
        use_fft: Use img_psf_conv or normal conv2d
    """
    # TODO: only convolve with PSFS that are necessary.
    img = tf.cast(img, dtype=tf.float32)
    input_shape = img.shape.as_list()

    zeros_tensor = tf.zeros_like(img, dtype=tf.float32)
    disc_depth_map = tf.tile(tf.cast(disc_depth_map, tf.int16),
                             multiples=[1, 1, 1, input_shape[3]])

    blurred_imgs = []
    for depth_idx, psf in enumerate(psfs):
        psf = tf.cast(psf, dtype=tf.float32)
        condition = tf.equal(disc_depth_map, tf.convert_to_tensor(depth_idx, tf.int16))
        blurred_img = img_psf_conv(img, psf)
        blurred_imgs.append(tf.where(condition,
                                     blurred_img,
                                     zeros_tensor))

    result = tf.reduce_sum(blurred_imgs, axis=0)
    return result


def get_spherical_wavefront_phase(resolution,
                                  physical_size,
                                  wave_lengths,
                                  source_distance):
    source_distance = tf.cast(source_distance, tf.float64)
    physical_size = tf.cast(physical_size, tf.float64)
    wave_lengths = tf.cast(wave_lengths, tf.float64)

    N, M = resolution
    [x, y] = np.mgrid[-N // 2:N // 2,
             -M // 2:M // 2].astype(np.float64)

    x = x / N * physical_size
    y = y / M * physical_size

    # Assume distance to source is approx. constant over wave
    curvature = tf.sqrt(x ** 2 + y ** 2 + source_distance ** 2)
    wave_nos = 2. * np.pi / wave_lengths

    phase_shifts = compl_exp_tf(wave_nos * curvature)
    phase_shifts = tf.expand_dims(tf.expand_dims(phase_shifts, 0), -1)
    return phase_shifts


def least_common_multiple(a, b):
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0


def area_downsampling_tf(input_image, target_side_length):
    input_shape = input_image.shape.as_list()
    input_image = tf.cast(input_image, tf.float32)

    if not input_shape[1] % target_side_length:
        factor = int(input_shape[1] / target_side_length)
        output_img = tf.nn.avg_pool(input_image,
                                    [1, factor, factor, 1],
                                    strides=[1, factor, factor, 1],
                                    padding="VALID")
    else:
        # We upsample the image and then average pool
        lcm_factor = least_common_multiple(target_side_length, input_shape[1]) / target_side_length

        if lcm_factor > 10:
            print(
                "Warning: area downsampling is very expensive and not precise if source and target wave length have a large least common multiple")
            upsample_factor = 10
        else:
            upsample_factor = int(lcm_factor)

        img_upsampled = tf.image.resize_nearest_neighbor(input_image,
                                                         size=2 * [upsample_factor * target_side_length])
        output_img = tf.nn.avg_pool(img_upsampled,
                                    [1, upsample_factor, upsample_factor, 1],
                                    strides=[1, upsample_factor, upsample_factor, 1],
                                    padding="VALID")

    return output_img


def get_intensities(input_field):
    return tf.square(tf.abs(input_field), name='intensities')


##################################
# Optical elements & Propagation
##################################

class Propagation(abc.ABC):
    def __init__(self,
                 input_shape,
                 distance,
                 discretization_size,
                 wave_lengths,
                 lens_index):
        self.input_shape = input_shape
        self.distance = distance
        self.wave_lengths = wave_lengths
        self.wave_nos = 2. * np.pi / wave_lengths
        self.discretization_size = discretization_size
        self.lens_index = lens_index

    @abc.abstractmethod
    def _propagate(self, input_field):
        """Propagate an input field through the medium
        """

    def __call__(self, input_field):
        return self._propagate(input_field)


class FresnelPropagation(Propagation):
    def _propagate(self, input_field):
        _, M_orig, N_orig, _ = self.input_shape
        # zero padding.
        
        Mpad = (M_orig // 4)
        Npad = (N_orig // 4)
        M = M_orig + 2 * Mpad
        N = N_orig + 2 * Npad
        padded_input_field = tf.pad(input_field,
                                    [[0, 0], [Mpad, Mpad], [Npad, Npad], [0, 0]])

        [x, y] = np.mgrid[-N // 2:N // 2,
                 -M // 2:M // 2]
        [x, y] = np.mgrid[-N // 2:N // 2,
                 -M // 2:M // 2]
        # Spatial frequency
        fx = x / (self.discretization_size * N)  # max frequency = 1/(2*pixel_size)
        fy = y / (self.discretization_size * M)

        # We need to ifftshift fx and fy here, because ifftshift doesn't exist in TF.
        fx = ifftshift(fx)
        fy = ifftshift(fy)

        fx = fx[None, :, :, None]
        fy = fy[None, :, :, None]

        squared_sum = np.square(fx) + np.square(fy)
        '''
        Mpad = 2 * M_orig
        Npad = 2 * N_orig
        M = M_orig + 2 * Mpad
        N = N_orig + 2 * Npad
        padded_input_field = tf.pad(input_field,
                                    [[0, 0], [Mpad, Mpad], [Npad, Npad], [0, 0]])

        [x, y] = np.mgrid[-N // 2:N // 2,
                 -M // 2:M // 2]
        # Spatial frequency
        fx = x / (self.discretization_size * N)  # max frequency = 1/(2*pixel_size)
        fy = y / (self.discretization_size * M)

        # We need to ifftshift fx and fy here, because ifftshift doesn't exist in TF.
        fx = ifftshift(fx)
        fy = ifftshift(fy)

        fx = fx[None, :, :, None]
        fy = fy[None, :, :, None]

        squared_sum = np.square(fx) + np.square(fy)
        '''
        # We create a non-trainable variable so that this computation can be reused
        # from call to call.
        if tf.contrib.framework.is_tensor(self.distance):
            tmp = np.float64(self.wave_lengths * np.pi * -1. * squared_sum)
            constant_exp_part_init = tf.constant_initializer(tmp)
            constant_exponent_part = tf.get_variable("Fresnel_kernel_constant_exponent_part"+self.lens_index,
                                                     initializer=constant_exp_part_init,
                                                     shape=padded_input_field.shape,
                                                     dtype=tf.float64,
                                                     trainable=False)

            H = compl_exp_tf(self.distance * constant_exponent_part, dtype=tf.complex64,
                             name='fresnel_kernel')
        else:  # Save some memory
            tmp = np.float64(self.wave_lengths * np.pi * -1. * squared_sum * self.distance)
            constant_exp_part_init = tf.constant_initializer(tmp)
            constant_exponent_part = tf.get_variable("Fresnel_kernel_constant_exponent_part"+self.lens_index,
                                                     initializer=constant_exp_part_init,
                                                     shape=padded_input_field.shape,
                                                     dtype=tf.float64,
                                                     trainable=False)

            H = compl_exp_tf(constant_exponent_part, dtype=tf.complex64,
                             name='fresnel_kernel')

        objFT = transp_fft2d(padded_input_field)
        out_field = transp_ifft2d(objFT * H)

        return out_field[:, Mpad:-Mpad, Npad:-Npad, :], input_field, H, objFT


class PhasePlate():
    def __init__(self,
                 wave_lengths,
                 height_map,
                 refractive_idcs,
                 height_tolerance=None,
                 lateral_tolerance=None):
        self.wave_lengths = wave_lengths
        self.height_map = height_map
        self.refractive_idcs = refractive_idcs
        self.height_tolerance = height_tolerance
        self.lateral_tolerance = lateral_tolerance

        self._build()

    def _build(self):
        # Add manufacturing tolerances in the form of height map noise
        if self.height_tolerance is not None:
            self.height_map += tf.random_uniform(shape=self.height_map.shape,
                                                 minval=-self.height_tolerance,
                                                 maxval=self.height_tolerance,
                                                 dtype=self.height_map.dtype)
            print("Phase plate with manufacturing tolerance %0.2e" % self.height_tolerance)

        self.phase_shifts = phaseshifts_from_height_map(self.height_map,
                                                        self.wave_lengths,
                                                        self.refractive_idcs)

    def __call__(self, input_field):
        input_field = tf.cast(input_field, tf.complex64)
        return tf.multiply(input_field, self.phase_shifts, name='phase_plate_shift')


def propagate_exact(input_field,
                    distance,
                    input_sample_interval,
                    wave_lengths):
    _, M_orig, N_orig, _ = input_field.shape.as_list()
    # zero padding.
    Mpad = M_orig // 4
    Npad = N_orig // 4
    M = M_orig + 2 * Mpad
    N = N_orig + 2 * Npad
    padded_input_field = tf.pad(input_field,
                                [[0, 0], [Mpad, Mpad], [Npad, Npad], [0, 0]])

    [x, y] = np.mgrid[-N // 2:N // 2,
             -M // 2:M // 2]

    # Spatial frequency
    fx = x / (input_sample_interval * N)  # max frequency = 1/(2*pixel_size)
    fy = y / (input_sample_interval * M)

    # We need to ifftshift fx and fy here, because ifftshift doesn't exist in TF.
    fx = ifftshift(fx)
    fy = ifftshift(fy)

    fx = fx[None, :, :, None]
    fy = fy[None, :, :, None]

    # We create a non-trainable variable so that this computation can be reused
    # from call to call.
    if tf.contrib.framework.is_tensor(distance):
        tmp = np.float64(
            2 * np.pi * (1 / wave_lengths) * np.sqrt(1. - (wave_lengths * fx) ** 2 - (wave_lengths * fy) ** 2))
        constant_exp_part_init = tf.constant_initializer(tmp)
        constant_exponent_part = tf.get_variable("Fresnel_kernel_constant_exponent_part",
                                                 initializer=constant_exp_part_init,
                                                 shape=padded_input_field.shape,
                                                 dtype=tf.float64,
                                                 trainable=False)

        H = compl_exp_tf(distance * constant_exponent_part, dtype=tf.complex64,
                         name='fresnel_kernel')
    else:  # Save some memory
        tmp = np.float64(
            2 * np.pi * (distance / wave_lengths) * np.sqrt(1. - (wave_lengths * fx) ** 2 - (wave_lengths * fy) ** 2))
        constant_exp_part_init = tf.constant_initializer(tmp)
        constant_exponent_part = tf.get_variable("Fresnel_kernel_constant_exponent_part",
                                                 initializer=constant_exp_part_init,
                                                 shape=padded_input_field.shape,
                                                 dtype=tf.float64,
                                                 trainable=False)

        H = compl_exp_tf(constant_exponent_part, dtype=tf.complex64,
                         name='fresnel_kernel')

    objFT = transp_fft2d(padded_input_field)
    out_field = transp_ifft2d(objFT * H)

    return out_field[:, Mpad:-Mpad, Npad:-Npad, :]


def propagate_fresnel(input_field,
                      distance,
                      sampling_interval,
                      wave_lengths,
                      lens_index):
    input_shape = input_field.shape.as_list()
    propagation = FresnelPropagation(input_shape,
                                     distance=distance,
                                     discretization_size=sampling_interval,
                                     wave_lengths=wave_lengths,
                                     lens_index=lens_index)
    return propagation(input_field)


def circular_aperture(input_field):
    input_shape = input_field.shape.as_list()
    [x, y] = np.mgrid[-input_shape[1] // 2: input_shape[1] // 2,
             -input_shape[2] // 2: input_shape[2] // 2].astype(np.float64)

    max_val = np.amax(x)

    r = np.sqrt(x ** 2 + y ** 2)[None, :, :, None]
    aperture = (r < max_val).astype(np.float64)
    return aperture * input_field


def height_map_element(input_field,
                       name,
                       wave_lengths,
                       refractive_idcs,
                       block_size=1,
                       height_map_initializer=None,
                       height_map_regularizer=None,
                       height_tolerance=None,  # Default height tolerance is 2 nm.
                       ):
    _, height, width, _ = input_field.shape.as_list()
    height_map_shape = [1, height // block_size, width // block_size, 1]

    if height_map_initializer is None:
        init_height_map_value = np.ones(shape=height_map_shape, dtype=np.float64) * 1e-4
        height_map_initializer = tf.constant_initializer(init_height_map_value)

    with tf.variable_scope(name, reuse=False):
        height_map_var = tf.get_variable(name="height_map_sqrt",
                                         shape=height_map_shape,
                                         dtype=tf.float64,
                                         trainable=True,
                                         initializer=height_map_initializer)

        height_map_full = tf.image.resize_images(height_map_var, height_map_shape[1:3],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        height_map = tf.square(height_map_full, name='height_map')

        if height_map_regularizer is not None:
            tf.contrib.layers.apply_regularization(height_map_regularizer, weights_list=[height_map])

        attach_summaries("Height_map", height_map, image=True, log_image=True)

    element = PhasePlate(wave_lengths=wave_lengths,
                         height_map=height_map,
                         refractive_idcs=refractive_idcs,
                         height_tolerance=height_tolerance)

    return element(input_field)


def fourier_element(input_field,
                    name,
                    wave_lengths,
                    refractive_idcs,
                    frequency_range=0.5,
                    height_map_regularizer=None,
                    height_tolerance=None,  # Default height tolerance is 2 nm.
                    ):
    _, height, width, _ = input_field.shape.as_list()
    height_map_shape = [1, height, width, 1]

    fourier_initializer = tf.zeros_initializer()

    with tf.variable_scope(name, reuse=False):
        fourier_vars_real = tf.get_variable('fourier_coeffs_real',
                                            shape=[1, int(height * frequency_range), int(width * frequency_range), 1],
                                            dtype=tf.float32,
                                            trainable=True,
                                            initializer=fourier_initializer)
        fourier_vars_cplx = tf.get_variable('fourier_coeffs_cmplx',
                                            shape=[1, int(height * frequency_range), int(width * frequency_range), 1],
                                            dtype=tf.float32,
                                            trainable=True,
                                            initializer=fourier_initializer)
        fourier_coeffs = tf.complex(fourier_vars_real, fourier_vars_cplx)
        attach_summaries("Fourier_coeffs", tf.abs(fourier_coeffs), image=True, log_image=False)
        padding_width = int((1 - frequency_range) * height) // 2
        fourier_coeffs_padded = tf.pad(fourier_coeffs,
                                       [[0, 0], [padding_width, padding_width], [padding_width, padding_width], [0, 0]])
        print(fourier_coeffs_padded.shape.as_list())
        height_map = tf.real(transp_ifft2d(ifftshift2d_tf(fourier_coeffs_padded)))

        if height_map_regularizer is not None:
            tf.contrib.layers.apply_regularization(height_map_regularizer, weights_list=[height_map])

        attach_summaries("Height_map", height_map, image=True, log_image=True)

    element = PhasePlate(wave_lengths=wave_lengths,
                         height_map=height_map,
                         refractive_idcs=refractive_idcs,
                         height_tolerance=height_tolerance)

    return element(input_field)


def zernike_element(input_field,
                    zernike_volume,
                    name,
                    wave_lengths,
                    refractive_idcs,
                    zernike_initializer=None,
                    height_map_regularizer=None,
                    height_tolerance=None,  # Default height tolerance is 2 nm.
                    zernike_scale=1e5,
                    ):
    _, height, width, _ = input_field.shape.as_list()
    height_map_shape = [1, height, width, 1]

    num_zernike_coeffs = zernike_volume.shape.as_list()[0]

    if zernike_initializer is None:
        zernike_initializer = tf.zeros_initializer()  # tf.random_normal_initializer(stddev=1e-6)

    with tf.variable_scope(name, reuse=False):
        zernike_coeffs = tf.get_variable('zernike_coeffs',
                                         shape=[num_zernike_coeffs, 1, 1],
                                         dtype=tf.float32,
                                         trainable=True,
                                         initializer=zernike_initializer)
        mask = np.ones([num_zernike_coeffs, 1, 1])
        mask[0] = 0.
        zernike_coeffs *= mask / zernike_scale

        for i in range(num_zernike_coeffs):
            tf.summary.scalar('zernike_coeff_%d' % i, tf.squeeze(zernike_coeffs[i, :, :]))

        height_map = tf.reduce_sum(zernike_coeffs * zernike_volume, axis=0)
        height_map = tf.expand_dims(tf.expand_dims(height_map, 0), -1, name='height_map')

        if height_map_regularizer is not None:
            tf.contrib.layers.apply_regularization(height_map_regularizer, weights_list=[height_map])

        height_map_summary = (height_map - tf.reduce_min(height_map)) / (
                    tf.reduce_max(height_map) - tf.reduce_min(height_map))
        attach_summaries("Height_map", height_map_summary, image=True, log_image=True)

    element = PhasePlate(wave_lengths=wave_lengths,
                         height_map=height_map,
                         refractive_idcs=refractive_idcs,
                         height_tolerance=height_tolerance)

    return element(input_field)


def gaussian_noise(image, stddev=0.001):
    dtype = image.dtype
    return image + tf.random_normal(image.shape, 0.0, stddev, dtype=dtype)


def get_vanilla_height_map(side_length,
                           height_map_regularizer=None,
                           name='height_map'):
    height_map_shape = [1, side_length, side_length, 1]

    init_height_map_value = np.ones(shape=height_map_shape, dtype=np.float64) * 1e-4
    height_map_initializer = tf.constant_initializer(init_height_map_value)

    with tf.variable_scope(name, reuse=False):
        height_map_sqrt = tf.get_variable(name="height_map_sqrt",
                                          shape=height_map_shape,
                                          dtype=tf.float64,
                                          trainable=True,
                                          initializer=height_map_initializer)
        height_map = tf.square(height_map_sqrt, name='height_map')

        if height_map_regularizer is not None:
            tf.contrib.layers.apply_regularization(height_map_regularizer, weights_list=[height_map])

        attach_summaries("Height_map", height_map, image=True, log_image=True)

        return tf.cast(height_map, tf.float64)


def get_fourier_height_map(side_length,
                           frequency_range=0.5,
                           height_map_regularizer=None,
                           name='fourier_height_map'):
    height_map_shape = [1, side_length, side_length, 1]

    fourier_initializer = tf.zeros_initializer()

    with tf.variable_scope(name, reuse=False):
        fourier_vars_real = tf.get_variable('fourier_coeffs_real',
                                            shape=[1, int(side_length * frequency_range),
                                                   int(side_length * frequency_range), 1],
                                            dtype=tf.float32,
                                            trainable=True,
                                            initializer=fourier_initializer)
        fourier_vars_cplx = tf.get_variable('fourier_coeffs_cmplx',
                                            shape=[1, int(side_length * frequency_range),
                                                   int(side_length * frequency_range), 1],
                                            dtype=tf.float32,
                                            trainable=True,
                                            initializer=fourier_initializer)
        fourier_coeffs = tf.complex(fourier_vars_real, fourier_vars_cplx)
        attach_summaries("Fourier_coeffs", tf.abs(fourier_coeffs), image=True, log_image=False)
        padding_width = int((1 - frequency_range) * side_length) // 2
        fourier_coeffs_padded = tf.pad(fourier_coeffs,
                                       [[0, 0], [padding_width, padding_width], [padding_width, padding_width], [0, 0]])
        print(fourier_coeffs_padded.shape.as_list())
        height_map = tf.real(transp_ifft2d(ifftshift2d_tf(fourier_coeffs_padded)))

        if height_map_regularizer is not None:
            tf.contrib.layers.apply_regularization(height_map_regularizer, weights_list=[height_map])

        attach_summaries("Height_map", height_map, image=True, log_image=True)
        return height_map


class SingleLensSetup():
    def __init__(self,
                 height_map,
                 wave_resolution,
                 wave_lengths,
                 sensor_distance,
                 sensor_resolution,
                 input_sample_interval,
                 refractive_idcs,
                 height_tolerance,
                 noise_model=gaussian_noise,
                 psf_resolution=None,
                 target_distance=None,
                 use_planar_incidence=True,
                 upsample=True,
                 depth_bins=None):

        self.wave_lengths = wave_lengths
        self.refractive_idcs = refractive_idcs

        self.wave_resolution = wave_resolution
        if psf_resolution is None:
            psf_resolution = wave_resolution
        self.psf_resolution = psf_resolution

        self.sensor_distance = sensor_distance
        self.noise_model = noise_model
        self.sensor_resolution = sensor_resolution
        self.input_sample_interval = input_sample_interval

        self.use_planar_incidence = use_planar_incidence
        self.upsample = upsample
        self.target_distance = target_distance
        self.depth_bins = depth_bins

        self.height_tolerance = height_tolerance
        self.height_map = height_map

        self.physical_size = float(self.wave_resolution[0] * self.input_sample_interval)
        self.pixel_size = self.input_sample_interval * np.array(wave_resolution) / np.array(sensor_resolution)

        print("Physical size is %0.2e.\nWave resolution is %d." % (self.physical_size, self.wave_resolution[0]))

        self.optical_element = PhasePlate(wave_lengths=self.wave_lengths,
                                          height_map=self.height_map,
                                          refractive_idcs=self.refractive_idcs,
                                          height_tolerance=self.height_tolerance)
        self.get_psfs()

    def get_psfs(self):
        # Sort the point source distances in increasing order
        if self.use_planar_incidence:
            input_fields = [tf.ones(self.wave_resolution, dtype=tf.float32)[None, :, :, None]]
        else:
            distances = self.depth_bins

            if self.target_distance is not None:
                distances += [self.target_distance]

            N, M = self.wave_resolution
            [x, y] = np.mgrid[-N // 2:N // 2,
                     -M // 2:M // 2].astype(np.float64)

            x = x / N * self.physical_size
            y = y / M * self.physical_size

            squared_sum = x ** 2 + y ** 2

            wave_nos = 2. * np.pi / self.wave_lengths
            wave_nos = wave_nos.reshape([1, 1, 1, -1])

            input_fields = []
            for distance in distances:
                # Assume distance to source is approx. constant over wave
                curvature = tf.sqrt(squared_sum + tf.cast(distance, tf.float64) ** 2)
                curvature = tf.expand_dims(tf.expand_dims(curvature, 0), -1)

                spherical_wavefront = compl_exp_tf(wave_nos * curvature, dtype=tf.complex64)
                input_fields.append(spherical_wavefront)

        psfs = []
        with tf.variable_scope("Forward_model") as scope:
            for depth_idx, input_field in enumerate(input_fields):
                field = self.optical_element(input_field)
                field = circular_aperture(field)
                sensor_incident_field = propagate_fresnel(field,
                                                          distance=self.sensor_distance,
                                                          sampling_interval=self.input_sample_interval,
                                                          wave_lengths=self.wave_lengths)
                psf = get_intensities(sensor_incident_field)

                if not self.upsample:
                    psf = area_downsampling_tf(psf, self.psf_resolution[0])

                psf = tf.div(psf, tf.reduce_sum(psf, axis=[1, 2], keep_dims=True), name='psf_depth_idx_%d' % depth_idx)

                attach_summaries('PSF_depth_idx_%d' % depth_idx, psf, image=True, log_image=True)
                psfs.append(tf.transpose(psf, [1, 2, 0, 3]))  # (Height, width, 1, channels)
                scope.reuse_variables()

        if self.target_distance is not None:
            self.target_psf = psfs.pop()
            attach_summaries('target_psf', tf.transpose(self.target_psf, [2, 0, 1, 3]), image=True)

        self.psfs = psfs

    def get_sensor_img(self,
                       input_img,
                       noise_sigma=0.001,
                       depth_dependent=False,
                       depth_map=None,
                       otfs=None):
        """"""
        # Upsample input_img to match wave resolution.
        if self.upsample:
            print("Images are upsampled to wave resolution")
            input_img = tf.image.resize_images(input_img, self.wave_resolution,
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        else:
            print("Images are not upsampled to wave resolution")

        if depth_dependent:
            if self.upsample:
                depth_map = tf.image.resize_images(depth_map, self.wave_resolution,
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            sensor_img = depth_dep_convolution(input_img, self.psfs, disc_depth_map=depth_map)
        else:
            sensor_img = img_psf_conv(input_img, self.psfs[0], otf=otfs)

        # Down sample measured image to match sensor resolution.
        if self.upsample:
            sensor_img = area_downsampling_tf(sensor_img, self.sensor_resolution[0])
        noisy_img = self.noise_model(sensor_img, noise_sigma)

        # print("Additive noise of %0.2e"%noise_sigma)
        attach_summaries("Sensor_img", noisy_img, image=True, log_image=False)

        return noisy_img


class ZernikeSystem():
    def __init__(self,
                 zernike_volume,
                 wave_resolution,
                 wave_lengths,
                 sensor_distance,
                 sensor_resolution,
                 input_sample_interval,
                 refractive_idcs,
                 height_tolerance,
                 focal_length,
                 lens_index,
                 target_distance=None,
                 upsample=True,
                 depth_bins=None):
        '''Simulates a one-lens system with a zernike-parameterized lens.

        :param zernike_volume: Zernike basis functions.
                               Tensor of shape (num_basis_functions, wave_resolution[0], wave_resolution[1]).
        :param wave_resolution: Resolution of the simulated wavefront. Shape wave_resolution.
        :param wave_lengths: Wavelengths to be simulated. Shape (num_wavelengths).
        :param sensor_distance: Distance of sensor to optical element.
        :param sensor_resolution: Resolution of simulated sensor.
        :param input_sample_interval: Sampling interval of aperture. Scalar.
        :param refractive_idcs: Refractive idcs of simulated material at wave_lengths.
        :param height_tolerance: Manufacturing tolerance of element. Adds the respective level of noise to be robust to
                                 manufacturing imperfections.
        :param target_distance: Allows to define the depth of a PSF that will *always* be evaluated. That can then be
                                used for instance for EDOF deconvolution.
        :param upsample: Whether the image should be upsampled to the PSF resolution or the PSF should be downsampled
                         to the sensor resolution.
        :param depth_bins: Depths at which PSFs should be simulated.
        '''

        self.sensor_distance = sensor_distance
        self.zernike_volume = zernike_volume
        self.wave_resolution = wave_resolution
        self.wave_lengths = wave_lengths
        self.depth_bins = depth_bins
        self.sensor_resolution = sensor_resolution
        self.upsample = upsample
        self.target_distance = target_distance
        self.zernike_volume = zernike_volume
        self.height_tolerance = height_tolerance
        self.input_sample_interval = input_sample_interval
        self.refractive_idcs = refractive_idcs
        self.focal_length = focal_length
        self.psf_resolution = self.sensor_resolution
        self.lens_index = lens_index

        self.physical_size = float(self.wave_resolution[0] * self.input_sample_interval)

        print("Physical size is %0.2e.\nWave resolution is %d." % (self.physical_size, self.wave_resolution[0]))

        self._build_height_map()
        self._get_psfs()

    def _build_height_map(self):
        '''Generates a zernike height map for optimization (residing in self.element after function call.)

        :return: None.
        '''
        #num_zernike_coeffs = self.zernike_volume.shape.as_list()[0]
        num_zernike_coeffs = 131
        zernike_inits = np.zeros((num_zernike_coeffs, 1, 1))
        
        # zernike_inits[3] = -51  # This sets the defocus value to approximately focus the image for a distance of 1m.
        # zernike_coef = [655.99451,-0.082540132,-0.094136864,65.714363,-0.12399390,-93.080322,-0.13301453,-0.11662229,-0.053188119,0.085961781,0.017194375,-0.21280862,-0.15987100,0.29557639,0.023300972,-0.14261581,-0.16267055,0.10510125,-0.065008678,0.0050166887,0.030329488,-0.17516507,-0.18879916,0.025095146,0.027527144,0.072598338,0.0035312974,-0.010586882,-0.18745133,-0.16432837,-0.074850783,0.12107035,0.034875765,0.0057113869,-0.00078142196,-0.00060794409,-0.19918774,0.029866653,-0.21353021,0.080547795,0.031147761,-0.010501395,0.0038932560,-0.010088747,0.0041283588,-0.18322222,-0.20902568,0.13494404,-0.083377086,0.0062688501,0.038782842,-0.00061002478,-0.00090593856,-0.021026708,-0.014226323,-0.21957545,-0.23531045,0.032957267,0.034345459,0.088613838,0.0041519096,-0.011545030,0.0045765932,-0.011106440,-0.035195641,-0.0019005949,-0.22823939,-0.20003819,-0.090922587,0.14726669,0.042204671,0.0067110709,-0.0010351200,-0.00057458383,-0.015535472,-0.022988290,-0.023257110,0.026534321,-0.23786779,0.035734612,-0.25482517,0.095781729,0.037220895,-0.012479761,0.0043151104,-0.011999630,0.0049915747,-0.0020645766,-0.038173798,0.026036970,0.0054546213,-0.21521042,-0.24558850,0.15835743,-0.097683035,0.0070503079,0.045229953,-0.00050144887,-0.0011711247,-0.024772078,-0.016719129,0.028584156,-0.025046518,0.0045463257,0.012872583,-0.25446624,-0.27249640,0.038268019,0.039835598,0.10219605,0.0043882700,-0.013316706,0.0053825816,-0.012796325,-0.040894955,-0.0022157466,0.0058499631,0.027859371,0.0015666845,-0.0036539705,-0.26139045,-0.22901748,-0.10378188,0.16841935,0.047915198,0.0072933105,-0.0013193444,-0.00039173698,-0.017798547,-0.026411390,-0.026684826]
        # zernike_coef = [192.42664,-0.028661937,-0.028658520,32.984501,-0.040234905,9.0687344e-08,-0.040493112,-0.040497959,-0.022584910,0.022584738,0.063208699,-3.6190283e-07,-0.051876701,0.019633725,-7.7301490e-07,-0.049525917,-0.049519900,0.027609732,-0.027610024,0.0057569039,0.0057588648,-0.056384869,-0.061263531,-1.5947577e-07,-5.8441481e-07,0.023164669,0.0011789283,1.9834388e-06,-0.057061601,-0.057068460,-0.031799220,0.031798888,0.0066128233,0.0066104988,-1.1780669e-05,8.3798759e-06,-0.064616844,-1.0357615e-07,-0.069288447,0.026166424,-7.3412440e-07,1.6000882e-06,0.0013007505,-0.0032910858,5.7947477e-07,-0.063633442,-0.063625813,0.035434578,-0.035434883,0.0073377239,0.0073403590,2.5197525e-05,-2.9238578e-05,-0.0057244045,-0.0057184678,-0.071234822,-0.076355994,-1.4495083e-07,-3.1268954e-07,0.028790237,0.0013884323,7.1540921e-07,1.2406703e-06,-0.0036198958,-0.011416271,-8.1354716e-07,-0.069470122,-0.069478519,-0.038659643,0.038659260,0.0079700965,0.0079672504,-5.4966160e-05,5.1439667e-05,-0.0062490697,-0.0062555633,-0.0080818236,0.0080678659,-0.077169396,-2.8247126e-07,-0.082688458,0.031118933,-1.6031848e-07,9.5883642e-08,0.0014449655,-0.0039109662,3.7963716e-07,-9.8485532e-07,-0.012382205,0.0084472531,-1.8166804e-07,-0.074754320,-0.074745275,0.041556571,-0.041556962,0.0085155005,0.0085186493,8.2822451e-05,-8.6034481e-05,-0.0067374129,-0.0067304182,0.0086899484,-0.0087049659,0.0028274141,0.0028314725,-0.082554303,-0.088422433,-3.4888618e-07,6.5757911e-07,0.033203062,0.0014718072,-6.1923458e-07,-5.7176084e-07,-0.0041708876,-0.013264789,-4.4479697e-07,1.4559114e-06,0.0090382304,0.00051796099,8.7337577e-07,-0.079548001,-0.079557613,-0.044179961,0.044179592,0.0089961234,0.0089927651,-0.00012633635,0.00012206868,-0.0071713738,-0.0071788519,-0.0092759598]
        # for i in range(131):
        #      zernike_inits[i] = zernike_coef[i]
        
        zernike_initializer = tf.constant_initializer(zernike_inits)

        self.zernike_coeffs = tf.get_variable('zernike_coeffs'+self.lens_index,
                                              shape=[num_zernike_coeffs, 1, 1],
                                              dtype=tf.float32,
                                              trainable=True,
                                              initializer=zernike_initializer)
        tf.add_to_collection('Optical_Variables', self.zernike_coeffs)
        
        for i in range(num_zernike_coeffs):
            tf.summary.scalar('zernike_coeff_%d' % i, tf.squeeze(self.zernike_coeffs[i, :, :]))

        self.height_map = tf.reduce_sum(self.zernike_coeffs * self.zernike_volume, axis=0)
        self.height_map = tf.expand_dims(tf.expand_dims(self.height_map, 0), -1, name='height_map')

        attach_summaries("Height_map", self.height_map, image=True, log_image=False)

        self.element = PhasePlate(wave_lengths=self.wave_lengths,
                                  height_map=self.height_map,
                                  refractive_idcs=self.refractive_idcs,
                                  height_tolerance=self.height_tolerance)
    
    
    def typical_lens(self, input_field, coordinate, input_focal_length):

        focal_length = input_focal_length
        focal_length_initializer = tf.constant_initializer(focal_length)
        wavelength = tf.reshape(self.wave_lengths,[1,1,1,-1])
        focal_length = tf.get_variable('focal_length'+self.lens_index,
                                              shape=wavelength.shape[3],
                                              dtype=tf.float64,
                                              trainable=False,
                                              initializer=focal_length_initializer)
        # wave number
        focal_length = tf.reshape(focal_length,[1,1,1,-1])
        wave_lengths = self.wave_lengths
        wave_nos = 2. * np.pi / wave_lengths
        wave_nos = wave_nos.reshape([1, 1, 1, -1])
        # phase delay indiced by height field
        coordinate = tf.expand_dims(tf.expand_dims(coordinate, 0), -1)
        phi = (-1 * wave_nos * coordinate)/(2 * focal_length)
        phase = compl_exp_tf(phi)
        
        return tf.multiply(input_field, phase, name='phase_plate_shift'), focal_length
    
    def _get_psfs(self):
        '''Builds the graph to generate psfs for depths in self.depth_bins, residing in self.psfs after function call.

        :return: None.
        '''
        # Sort the point source distances in increasing order

        #distances = self.depth_bins
        distances = [50e-3]
        if self.target_distance is not None:
            distances += [self.target_distance]

        N, M = self.wave_resolution
        [x, y] = np.mgrid[-N // 2:N // 2,
                 -M // 2:M // 2].astype(np.float64)

        x = x / N * self.physical_size
        y = y / M * self.physical_size

        squared_sum = x ** 2 + y ** 2

        wave_nos = 2. * np.pi / self.wave_lengths
        wave_nos = wave_nos.reshape([1, 1, 1, -1])

        input_fields = []
        for distance in distances:
            # Assume distance to source is approx. constant over wave
            curvature = tf.sqrt(squared_sum + tf.cast(distance, tf.float64) ** 2)
            #curvature = (squared_sum)/(2 * tf.cast(distance, tf.float64))
            curvature = tf.expand_dims(tf.expand_dims(curvature, 0), -1)

            spherical_wavefront = compl_exp_tf(wave_nos * curvature, dtype=tf.complex64)
            input_fields.append(spherical_wavefront)

        psfs = []
        with tf.variable_scope("Forward_model") as scope:
            for depth_idx, input_field in enumerate(input_fields):
                #input_field = tf.ones(self.wave_resolution, dtype=tf.complex64)[None, :, :, None]
                field, focal_length = self.typical_lens(input_field,
                                         squared_sum,self.focal_length)
                field = self.element(field)
                #distance_phi = compl_exp_tf(-1 * wave_nos * self.sensor_distance)
                #field = tf.multiply(field, distance_phi)
                field = circular_aperture(field)
                sensor_incident_field, field, H, objFT = propagate_fresnel(field,
                                                          distance=self.sensor_distance,
                                                          sampling_interval=self.input_sample_interval,
                                                          wave_lengths=self.wave_lengths,
                                                          lens_index=self.lens_index)
                psf = sensor_incident_field
                #amp = tf.cast(tf.abs(psf), tf.complex64)
                intensity_psf = get_intensities(psf)
                normalized = tf.reduce_sum(intensity_psf, axis=[1, 2], keep_dims=True)
                normalized = tf.cast(tf.sqrt(normalized), tf.complex64)
                if self.upsample:
                    psf = area_downsampling_tf(psf, self.psf_resolution[0])

                psf = tf.div(psf, normalized, name='psf_depth_idx_%d' % depth_idx)

                #attach_summaries('PSF_depth_idx_%d' % depth_idx, psf, image=True, log_image=True)
                psfs.append(tf.transpose(psf, [1, 2, 0, 3]))  # (Height, width, 1, channels)
                scope.reuse_variables()

        if self.target_distance is not None:
            self.target_psf = psfs.pop()
            #attach_summaries('target_psf', tf.transpose(self.target_psf, [2, 0, 1, 3]), image=True)

        self.psfs, self.objFT, self.H = psfs, objFT, H
        self.field, self.intensity_psf, self.focal = field, psf, focal_length

    def get_sensor_img(self,
                       input_img,
                       noise_sigma,
                       depth_dependent=False,
                       depth_map=None):
        """Calculates the sensor image.

        :param input_img: Imaged scene.
        :param noise_sigma: Sigma of gaussian sensor noise. Scalar.
        :param depth_dependent: Whether to use a depth_map.
        :param depth_map: A discretized depth map, where every pixel is an index into self.depth_bins.
                          Shape (batch_size, self.sensor_resolution[0], self.sensor_resolution[1])
        :return: Sensor image.
        """
        # Upsample input_img to match wave resolution.
        if self.upsample:
            print("Images are upsampled to wave resolution")
            input_img = tf.image.resize_images(input_img, self.wave_resolution,
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        else:
            print("Images are not upsampled to wave resolution")

        if depth_dependent:
            if self.upsample:
                depth_map = tf.image.resize_images(depth_map, self.wave_resolution,
                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            sensor_img = depth_dep_convolution(input_img, self.psfs, disc_depth_map=depth_map)
        else:
            sensor_img = img_psf_conv(input_img, self.target_psf)
            sensor_img = tf.abs(sensor_img)
            input_energy = tf.reduce_sum(tf.abs(input_img), axis=[1,2], keep_dims=True)
            output_energy = tf.reduce_sum(sensor_img, axis=[1,2], keep_dims=True)
            normalize_weight = input_energy / output_energy
            sensor_img = sensor_img * normalize_weight
            sensor_img = get_intensities(sensor_img)
            
            
            #div_psf = tf.transpose(self.intensity_psf, [2, 0, 1, 3])
            #sensor_img = tf.div(sensor_img, tf.reduce_sum(self.intensity_psf, axis=[1, 2], keep_dims=True), name='psf')


        # Downsample measured image to match sensor resolution.
        if self.upsample:
            sensor_img = area_downsampling_tf(sensor_img, self.sensor_resolution[0])

        return sensor_img
