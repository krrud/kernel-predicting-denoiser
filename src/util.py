import numpy as np
import Imath
import OpenEXR
import minexr
import random
import os
import sys
import math


def albedo_correction(render, albedo, emission, specular, remove=True):
    """ Divides/multiplies by albedo, and subtracts/adds emission """

    # Add neutral depth padding to albedo, emission, and specular
    filler = render.shape[2] - albedo.shape[2]
    ones = np.ones((render.shape[0], render.shape[1], filler))
    zeros = np.zeros((render.shape[0], render.shape[1], filler))
    a = np.dstack((albedo, ones))
    e = np.dstack((emission, zeros))
    s = np.dstack((specular, zeros))

    if remove:
        # Subtract emission and spec, then divide by albedo
        corrected = np.divide(render-e-s, a, out=np.zeros_like(render), where=a != 0)
        return corrected.astype(np.float32)

    else:
        # Multiply by albedo and add emission back in
        corrected = render * a + e
        return corrected.astype(np.float32)


def spec_log(rgb):
    """ Log transformation to help with HDR specular inputs """

    log_r = np.log1p(rgb[:, :, 0])
    log_g = np.log1p(rgb[:, :, 1])
    log_b = np.log1p(rgb[:, :, 2])
    log_rgb = np.dstack((log_r, log_g, log_b))

    return log_rgb


def spec_inverse_log(rgb):
    """ Inverse log transform to expand back to HDR """

    ilog_r = np.expm1(rgb[:, :, 0])
    ilog_g = np.expm1(rgb[:, :, 1])
    ilog_b = np.expm1(rgb[:, :, 2])
    inverse_log_rgb = np.dstack((ilog_r, ilog_g, ilog_b))

    return inverse_log_rgb


def minmax(arr):
    """ Channel-wise minmax normalization """

    min_vals = arr.min(axis=tuple(range(arr.ndim-1)))
    max_vals = arr.max(axis=tuple(range(arr.ndim-1)))

    normalized = np.zeros_like(arr)
    for i in range(arr.shape[-1]):
        normalized[..., i] = np.interp(arr[..., i], (min_vals[i], max_vals[i]), (0, 1))

    return normalized


def generate_grad(channels):
    """ Creates grads for each axis """

    grad_x, grad_y = [], []

    if channels.shape[-1] == 3:
        # Generate grads for RGB input
        grad_x = np.dstack((
            np.gradient(channels[:, :, 0])[0],
            np.gradient(channels[:, :, 1])[0],
            np.gradient(channels[:, :, 2])[0]))
        grad_y = np.dstack((
            np.gradient(channels[:, :, 0])[1],
            np.gradient(channels[:, :, 1])[1],
            np.gradient(channels[:, :, 2])[1]))

        grad_x, grad_y = minmax(grad_x), minmax(grad_y)

    else:
        # Generate grad for grayscale
        grad_x, grad_y = np.gradient(channels[:, :, 0])
        grad_x, grad_y = minmax(grad_x), minmax(grad_y)

    return grad_x, grad_y


def output_rgb(channels, filepath):
    """ Outputs an RGB exr to the given path """

    r, g, b = channels[:, :, 0], channels[:, :, 1], channels[:, :, 2]

    header = OpenEXR.Header(channels.shape[0], channels.shape[1])
    header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    output_file = OpenEXR.OutputFile(filepath, header)
    pixel_data = {'R': r.tobytes(), 'G': g.tobytes(),'B': b.tobytes()}
    output_file.writePixels(pixel_data)
    output_file.close()


def luminance(rgb):
    return .2126*rgb[:, :, 0] + .7152 * rgb[:, :, 1] + .0722*rgb[:, :, 2]


def read_exr(file, filepath):
    """ Reads and normalizes channels from exr """
    # Load exr file
    reader = minexr.load(file)

    # Extract primary channels
    rgb = reader.select(['R', 'G', 'B']).astype(np.float32)
    specular = reader.select(
        ['specular.R', 'specular.G', 'specular.B']).astype(np.float32)
    emission = reader.select(
        ['emission.R', 'emission.G', 'emission.B']).astype(np.float32)
    
    # Extract albedo and alpha - will pull from util file if available
    albedo, alpha  = None, None
    base, extension = os.path.splitext(filepath)
    util_file =  base + '_util' + extension
    if os.path.exists(util_file):
        with open(util_file, 'rb') as util:
            # Extract the high sample util passes
            print(f'UTIL FOUND: {util_file}')
            util_reader = minexr.load(util)
            alpha = util_reader.select(['A']).astype(np.float32)
            albedo = util_reader.select(
                ['albedo.R', 'albedo.G', 'albedo.B']).astype(np.float32)
    else:
        # Otherwise use the low sample passes from the noisy render
        alpha = reader.select(['A']).astype(np.float32)
        albedo = reader.select(
            ['albedo.R', 'albedo.G', 'albedo.B']).astype(np.float32)

    # Preprocess primary channels
    diffuse = albedo_correction(rgb, albedo, emission, specular, True)
    specular = spec_log(specular)


    # Extract diffuse and specular variance - apply transforms
    spec_var = reader.select(['specular_1.R', 'specular_1.G', 'specular_1.B']).astype(np.float32)
    diff_var = reader.select(['beauty_1.R', 'beauty_1.G', 'beauty_1.B']).astype(np.float32)

    diff_var = diff_var - spec_var
    spec_var = spec_var * specular ** 2
    diff_var = diff_var * (albedo + 1e-8) ** 2

    spec_var = spec_var / np.max(spec_var)
    diff_var = diff_var / np.max(diff_var)
    diff_var = luminance(diff_var)
    spec_var = luminance(spec_var)

    # Extract albedo variance
    albedo_var = reader.select(['albedo_1.R', 'albedo_1.G', 'albedo_1.B']).astype(np.float32)
    albedo_var = albedo_var / np.max(albedo_var)
    albedo_var = luminance(albedo_var)

    # Extract normal and normal variance
    normal = reader.select(['N.X', 'N.Y', 'N.Z']).astype(np.float32)
    normal_var = reader.select(['N_1.X', 'N_1.Y', 'N_1.Z']).astype(np.float32)
    normal_var = normal_var / np.max(normal_var)
    normal_var = luminance(normal_var)

    # Extract depth and depth variance
    depth = reader.select(['Z']).astype(np.float32)
    depth = depth / np.max(depth)
    depth_var = reader.select(['Z_1']).astype(np.float32)
    depth_var = depth_var / np.max(depth_var)

    # Generate grads for both axis
    diff_grad_x, diff_grad_y = generate_grad(diffuse)
    spec_grad_x, spec_grad_y = generate_grad(specular)
    albedo_grad_x, albedo_grad_y = generate_grad(albedo)
    normal_grad_x, normal_grad_y = generate_grad(normal)
    depth_grad_x, depth_grad_y = generate_grad(depth)

    # Depth stacked arrays
    diff_stack = np.dstack((
                            diffuse,
                            albedo, normal, depth, 
                            albedo_var, normal_var, depth_var, diff_var,
                            diff_grad_x, diff_grad_y,
                            albedo_grad_x, albedo_grad_y,
                            normal_grad_x, normal_grad_y,
                            depth_grad_x, depth_grad_y
                            ))
    spec_stack = np.dstack((
                            specular,
                            albedo, normal, depth,                            
                            albedo_var, normal_var, depth_var, spec_var,
                            spec_grad_x, spec_grad_y,
                            albedo_grad_x, albedo_grad_y,
                            normal_grad_x, normal_grad_y,
                            depth_grad_x, depth_grad_y
                            ))

    return diff_stack, spec_stack, albedo, emission, alpha


def read_openexr(file, filepath):

    """ Mirror of read_exr but utilizes openexr to load the file
        Much slower, but can read compressed and tiled exr files """
    
    # Define exr header and file
    exr_file = OpenEXR.InputFile(file)
    header = exr_file.header()
    channels = header['channels']
    width = header['dataWindow'].max.x - header['dataWindow'].min.x + 1
    height = header['dataWindow'].max.y - header['dataWindow'].min.y + 1
    exr_data = {}

    # Extract channel data
    for channel_name in channels.keys():
        channel_data_raw = exr_file.channel(
            channel_name, Imath.PixelType(Imath.PixelType.FLOAT))
        exr_data[channel_name] = np.frombuffer(
            channel_data_raw, dtype=np.float32).reshape((height, width))

    # Extract primary channels
    rgb = np.dstack((exr_data['R'], exr_data['G'],exr_data['B'])).astype(np.float32)
    specular = np.dstack(
        (exr_data['specular.R'], exr_data['specular.G'], exr_data['specular.B'])).astype(np.float32)
    emission = np.dstack(
        (exr_data['emission.R'], exr_data['emission.G'], exr_data['emission.B'])).astype(np.float32)
    
    albedo, alpha  = None, None
    base, extension = os.path.splitext(filepath)
    util_file =  base + '_util' + extension
    if os.path.exists(util_file):
        with open(util_file, 'rb') as util:
            # Extract high sample utility passes if available
            print(f'UTIL FOUND: {util_file}')
            util = OpenEXR.InputFile(util)
            header = exr_file.header()
            channels = header['channels']
            width = header['dataWindow'].max.x - header['dataWindow'].min.x + 1
            height = header['dataWindow'].max.y - header['dataWindow'].min.y + 1
            alpha_raw = util.channel('A', Imath.PixelType(Imath.PixelType.FLOAT))
            alpha = np.frombuffer(
                alpha_raw, dtype=np.float32).reshape((height, width))
            albedo_r = np.frombuffer(util.channel('albedo.R', Imath.PixelType(
                Imath.PixelType.FLOAT)), dtype=np.float32).reshape((height, width))
            albedo_g = np.frombuffer(util.channel('albedo.G', Imath.PixelType(
                Imath.PixelType.FLOAT)), dtype=np.float32).reshape((height, width))
            albedo_b = np.frombuffer(util.channel('albedo.B', Imath.PixelType(
                Imath.PixelType.FLOAT)), dtype=np.float32).reshape((height, width))
            albedo = np.dstack((albedo_r, albedo_g, albedo_b))
        
    else:
        # Use low sample utility passes from noisy render
        alpha = exr_data['A'].astype(np.float32)
        albedo = np.dstack(
            (exr_data['albedo.R'], exr_data['albedo.G'], exr_data['albedo.B'])).astype(np.float32)

    diffuse = albedo_correction(rgb, albedo, emission, specular, True)
    specular = spec_log(specular)

    # Extract diffuse and specular variance - apply transforms
    diff_var = np.dstack((
        exr_data['beauty_1.R'], 
        exr_data['beauty_1.G'], 
        exr_data['beauty_1.B']
        )).astype(np.float32)
    
    spec_var = np.dstack((
        exr_data['specular_1.R'], 
        exr_data['specular_1.G'], 
        exr_data['specular_1.B']
        )).astype(np.float32)

    diff_var = diff_var - spec_var
    spec_var = spec_var * specular ** 2
    diff_var = diff_var * (albedo + 1e-8) ** 2

    spec_var = spec_var / np.max(spec_var)
    diff_var = diff_var / np.max(diff_var)

    diff_var = luminance(diff_var)
    spec_var = luminance(spec_var)

    # Extract albedo variance
    albedo_var = np.dstack((
        exr_data['albedo_1.R'], 
        exr_data['albedo_1.G'], 
        exr_data['albedo_1.B']
        )).astype(np.float32)
    albedo_var = albedo_var / np.max(albedo_var)
    albedo_var = luminance(albedo_var)

    # Extract normal and normal variance
    normal = np.dstack((exr_data['N.X'], exr_data['N.Y'], exr_data['N.Z'])).astype(np.float32)
    normal_var = np.dstack((exr_data['N_1.X'], exr_data['N_1.Y'], exr_data['N_1.Z'])).astype(np.float32)
    normal_var = normal_var / np.max(normal_var)
    normal_var = luminance(normal_var)

    # Extract depth and depth variance
    depth = np.expand_dims(exr_data['Z'].astype(np.float32), axis=-1)
    depth = depth / np.max(depth)
    depth_var = exr_data['Z_1'].astype(np.float32)
    depth_var = depth_var / np.max(depth_var)
    
    # Generate grads
    diff_grad_x, diff_grad_y = generate_grad(diffuse)
    spec_grad_x, spec_grad_y = generate_grad(specular)
    albedo_grad_x, albedo_grad_y = generate_grad(albedo)
    normal_grad_x, normal_grad_y = generate_grad(normal)
    depth_grad_x, depth_grad_y = generate_grad(depth)

    # Depth stacked arrays 
    diff_stack = np.dstack((
                            diffuse,
                            albedo, normal, depth, 
                            albedo_var, normal_var, depth_var, diff_var,
                            diff_grad_x, diff_grad_y,
                            albedo_grad_x, albedo_grad_y,
                            normal_grad_x, normal_grad_y,
                            depth_grad_x, depth_grad_y
                            ))
    spec_stack = np.dstack((
                            specular,
                            albedo, normal, depth,                            
                            albedo_var, normal_var, depth_var, spec_var,
                            spec_grad_x, spec_grad_y,
                            albedo_grad_x, albedo_grad_y,
                            normal_grad_x, normal_grad_y,
                            depth_grad_x, depth_grad_y
                            ))

    return diff_stack, spec_stack, albedo, emission, alpha


def build_data(path, samples=300, predict=False):
    """ Constructs a dictionary of numpy arrays from all renders within the path """

    # Initialize dictionary
    data = {
        "noisy_diff": [],
        "noisy_spec": [],
        "clean_diff": [],
        "clean_spec": [],
        "albedo": [],
        "emission": [],
        "alpha": [],
        "name": []
    }

    # Isolate valid exr's
    if os.path.isfile(path):
        ext = path.split('.')[-1]
        if ext == 'exr' and '_util.' not in path and '_denoised.' not in path:
                data["name"].append(path)

    else:
        for file in os.listdir(path):
            ext = file.split('.')[-1]
            if ext == 'exr' and '_util.' not in file and '_denoise.' not in file:
                if '4spp' in file or predict:
                    data["name"].append(path + file)

    # Randomly pick valid files to train on 
    if not predict:        
        data["name"] = random.sample(data["name"], int(samples/256))

    # Extract data
    for render in data["name"]:
        pad = ' ' * 35
        n = render.split('/')[-1]
        sys.stdout.write(f'\rPreprocessing {n}...' + pad)
        sys.stdout.flush()

        # Extract and preprocess noisy render data
        try:
            with open(render, 'rb') as noisy_exr:
                diffuse, specular, albedo, emission, alpha = read_exr(noisy_exr, render)
        except AssertionError:
            with open(render, 'rb') as noisy_exr:
                diffuse, specular, albedo, emission, alpha = read_openexr(noisy_exr, render)     
        except:
            raise Exception("Failed to load exr.")
                
        # Append data to dictionary
        data["noisy_diff"].append(diffuse)
        data["noisy_spec"].append(specular)
        data["albedo"].append(albedo)
        data["emission"].append(emission)
        data["alpha"].append(alpha)

        if not predict:    
            # Extract and preprocess ground truth render data
            clean_render = render.replace('4spp', '32spp') 
            cn = clean_render.split('/')[-1]        
            with open(clean_render, 'rb') as clean_exr:
                sys.stdout.write(f'\rPreprocessing {cn}...' + pad)
                sys.stdout.flush()
                try:
                    # Open with minexr which is very efficient
                    diffuse, specular, albedo, emission, alpha = read_exr(clean_exr, clean_render)
                except:
                    try:
                        # Open with OpenExr - capable of opening compressed exr's
                        diffuse, specular, albedo, emission = read_openexr(clean_exr, clean_render)
                    except ValueError:
                        raise Exception(f'Invalid file: {clean_render}')                
                data["clean_diff"].append(diffuse)
                data["clean_spec"].append(specular)

    return data


def augment_data(x, y, amount=0.3, shuffle=True):
    """ Augment with randomly rotated inputs """

    out_x, out_y = [], []

    # Augment x and y with random rotations - does not add data
    if shuffle:
        for i in range(len(x)):
            roll = random.random()
            if roll < 1/3:
                x[i], y[i] = np.rot90(x[i], k=1), np.rot90(y[i], k=1, axes=(0, 1))
            elif 1/3 <= roll < 2/3:
                x[i], y[i] = np.rot90(x[i], k=2), np.rot90(y[i], k=2, axes=(0, 1))
            else:
                x[i], y[i] = np.rot90(x[i], k=3), np.rot90(y[i], k=3, axes=(0, 1))

        out_x, out_y = x, y

    else:
        # Add randomly rotated samples to the dataset - this will increase the dataset's size
        if amount < 1:
            split = int(len(x) * amount)
            ax, ay = x[:split], y[:split]
            for i in range(split):
                roll = random.random()
                if roll < 1/3:
                    ax[i], ay[i] = np.rot90(ax[i], k=1), np.rot90(
                        ay[i], k=1, axes=(0, 1))
                elif 1/3 <= roll < 2/3:
                    ax[i], ay[i] = np.rot90(ax[i], k=2), np.rot90(
                        ay[i], k=2, axes=(0, 1))
                else:
                    ax[i], ay[i] = np.rot90(ax[i], k=3), np.rot90(
                        ay[i], k=3, axes=(0, 1))

        out_x, out_y = x + ax, y + ay

    return out_x, out_y


def generate_tiles(data, size: int=64):
    """ Slices image arrays into smaller chunks """

    image_tiles = []
    for render in data:
        tile_list = []
        for r in range(0, render.shape[0], size):
            for c in range(0, render.shape[1], size):
                tile = render[r:r + size, c:c + size, :]
                tile_list.append(tile)
        image_tiles.append(tile_list)

    return (image_tiles)


def rebuild_image(img_tiles: np.ndarray, size: int):
    """ Rebuilds image from tiles """

    depth = img_tiles[0].shape[2]
    rebuilt_array = np.empty((size[0], size[1], depth))
    tile_size = len(img_tiles[0])
    tile = 0
    for i in range(int(size[0] / len(img_tiles[0]))):
        for j in range(int(size[0] / len(img_tiles[0]))):
            x1, x2 = i * tile_size, (i + 1) * tile_size
            y1, y2 = j * tile_size, (j + 1) * tile_size
            rebuilt_array[x1:x2, y1:y2] = img_tiles[tile]
            tile += 1

    return rebuilt_array.astype(np.float32)


def output_denoised(diffuse, specular, albedo, emission, alpha, name):

    # Apply the inverse of the preprocessing transforms
    diff = albedo_correction(diffuse, albedo, emission, specular, False)
    spec = spec_inverse_log(specular)

    # Combined the denoised and corrected channels
    denoised = diff + spec

    # Extract output channels
    r, g, b, a = denoised[:, :, 0], denoised[:, :, 1], denoised[:, :, 2], alpha
    diff_r, diff_g, diff_b = diff[:, :, 0], diff[:, :, 1], diff[:, :, 2]
    spec_r, spec_g, spec_b = spec[:, :, 0], spec[:, :, 1], spec[:, :, 2]

    # Define output exr and header
    out_name = '.'.join(name.split('.')[:-1]) + '_denoised.exr'
    header = OpenEXR.Header(diff.shape[1], diff.shape[0])
    header['channels'] = {
                        'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'A': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'diffuse.R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'diffuse.G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'diffuse.B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'specular.R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'specular.G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'specular.B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                        }

    # Save to disk and close the file
    output_file = OpenEXR.OutputFile(out_name, header)
    pixel_data = {
                'R': r.tobytes(), 
                'G': g.tobytes(),
                'B': b.tobytes(), 
                'A': a.tobytes(),
                'diffuse.R': diff_r.tobytes(),
                'diffuse.G': diff_g.tobytes(),
                'diffuse.B': diff_b.tobytes(),
                'specular.R': spec_r.tobytes(),
                'specular.G': spec_g.tobytes(),
                'specular.B': spec_b.tobytes()
                }
    output_file.writePixels(pixel_data)
    output_file.close()


def data_prep(
        path: str, 
        channel: str, 
        samples: int, 
        augment: bool=False):
    
    """ Builds image arrays and splits them into training/validation sets
     
        path: Directory of exr's to preprocess for training
        channel: Channel to prepare the data for - diffuse or specular
        sampes: How many samples to gather from the dataset
        val: Whether or not this data is being prepped for use as validation data
        augment: If True - will augment the dataset with random rotations """

    # Generate data
    data = build_data(path, samples=samples, predict=False)

    # Define channel: diffuse or spec
    c = None
    if channel in ['diffuse', 'diff', 'd']:
        c = 'diff'
    elif channel in ['specular', 'spec', 's']:
        c = 'spec'
    else:
        raise ValueError(f"Invalid channel input: {channel}")

    # Generate tiles for x
    x_full = data['noisy_' + c]
    x_tiles = generate_tiles(x_full, 64)
    x = np.array([tile for render in x_tiles for tile in render])

    # Generate tiles for y
    y_full = data['clean_' + c]
    y_tiles = generate_tiles(y_full, 64)
    y = np.array([tile for render in y_tiles for tile in render])

    # Augment the data if enabled
    if augment:
        x, y = augment_data(x, y)

    # Isolate rgb and feature inputs
    x_rgb, x_feature = np.array_split(x, [3], axis=-1)
    y_rgb, y_feature = np.array_split(y, [3], axis=-1)

    return x_rgb, x_feature, y_rgb, y_feature


def denoise_image(
        diffuse: np.ndarray, 
        specular: np.ndarray, 
        diff_denoiser, 
        spec_denoiser, 
        patch_size: int=64, 
        overlap: int=16):

    """ Samples overlapping patches which are then denoised with the kpn model
        Denoised patches are recombined and averaged to create final denoised output 
        
        diffuse: Noisy diffuse array
        specular: Noisy specular array
        diff_denoiser: KPNNModel to denoise the diffuse channel
        spec_denoiser: KPNNModel to denoise the cpecular channel
        patch_size: Size of the patches to be fed to the model
        overlap: Number of pixels to overlap each denoising patch """
    
    # Resolve overlap
    overlap = int(max(min(overlap, 63), 0))

    # Calculate number of patches in each dimension
    h, w = diffuse.shape[:2]
    ph = math.ceil((h - patch_size) / (patch_size - overlap) + 1)
    pw = math.ceil((w - patch_size) / (patch_size - overlap) + 1)
    
    # Initialize denoised arrays
    denoised_diffuse = np.zeros_like(diffuse[:, :, :3])
    denoised_specular = np.zeros_like(specular[:, :, :3])
    patch_counts = np.zeros_like(diffuse[:, :, :3])
    diff_patches, spec_patches = [], []
    
    # Generate patches
    for i in range(ph):
        for j in range(pw):
            h1 = i * (patch_size - overlap)
            h2 = h1 + patch_size
            w1 = j * (patch_size - overlap)
            w2 = w1 + patch_size
            if h2 > h:
                h2 = h
                h1 = h2 - patch_size
            if w2 > w:
                w2 = w
                w1 = w2 - patch_size
            diff_patch = diffuse[h1:h2, w1:w2]
            spec_patch = specular[h1:h2, w1:w2]
            diff_patches.append(diff_patch)
            spec_patches.append(spec_patch)

    # Denoise the patches
    diff_patches = np.stack(diff_patches)
    spec_patches = np.stack(spec_patches)
    dc, df = diff_patches[:, :, :, :3], diff_patches[:, :, :, 3:]
    sc, sf = spec_patches[:, :, :, :3], spec_patches[:, :, :, 3:]
    denoised_diff_patches = diff_denoiser.predict([dc, df], verbose=0)
    denoised_spec_patches = spec_denoiser.predict([sc, sf], verbose=0)

    # Recombine denoised patches 
    for i in range(len(diff_patches)):
        h1 = i // pw * (patch_size - overlap)
        h2 = h1 + patch_size
        w1 = i % pw * (patch_size - overlap)
        w2 = w1 + patch_size
        if h2 > h:
            h2 = h
            h1 = h2 - patch_size
        if w2 > w:
            w2 = w
            w1 = w2 - patch_size
        denoised_diff_patch = denoised_diff_patches[i]
        denoised_spec_patch = denoised_spec_patches[i]
        denoised_diffuse[h1:h2, w1:w2] += denoised_diff_patch
        denoised_specular[h1:h2, w1:w2] += denoised_spec_patch
        patch_counts[h1:h2, w1:w2] += 1

    # Normalize against patch counts
    denoised_diffuse /= patch_counts
    denoised_specular /= patch_counts

    return denoised_diffuse, denoised_specular


