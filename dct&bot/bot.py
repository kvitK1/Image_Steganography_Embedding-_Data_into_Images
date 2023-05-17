from PIL import Image
import numpy as np
import bitstring
import telebot
import struct
import cv2
import json
# import io

#########################DCT part#########################
HORIZ_AXIS = 1
VERT_AXIS  = 0

JPEG_STD_LUM_QUANT_TABLE50 = np.asarray([
                                  [16, 11, 10, 16,  24, 40,   51,  61],
                                  [12, 12, 14, 19,  26, 58,   60,  55],
                                  [14, 13, 16, 24,  40, 57,   69,  56],
                                  [14, 17, 22, 29,  51, 87,   80,  62],
                                  [18, 22, 37, 56,  68, 109, 103,  77],
                                  [24, 36, 55, 64,  81, 104, 113,  92],
                                  [49, 64, 78, 87, 103, 121, 120, 101],
                                  [72, 92, 95, 98, 112, 100, 103,  99]
                                 ],dtype = np.float32)

def prepare_image(image_path, secret_image_path):
    image = Image.open(image_path)
    image = image.convert("L")
    secret_image = Image.open(secret_image_path)
    secret_image = secret_image.convert("L")

    image_array = np.array(image)
    secret_image = secret_image.convert("L")

def split_8x8_blocks(image):
    blocks = []
    for v_slice in np.vsplit(image, int(image.shape[0] / 8)):
        for h_slice in np.hsplit(v_slice, int(image.shape[1] / 8)):
            blocks.append(h_slice)
    return blocks

def stitch_8x8_blocks(pixel_num, block_segments):
    image_rows = []
    temp = []
    for i in range(len(block_segments)):
        if i > 0 and not(i % int(pixel_num / 8)):
            image_rows.append(temp)
            temp = [block_segments[i]]
        else:
            temp.append(block_segments[i])
    image_rows.append(temp)

    return np.block(image_rows)

class YCC_Image(object):
    def __init__(self, cover_image):
        self.height, self.width = cover_image.shape[:2]
        self.channels = [
                         split_8x8_blocks(cover_image[:,:,0]),
                         split_8x8_blocks(cover_image[:,:,1]),
                         split_8x8_blocks(cover_image[:,:,2]),
        ]

def embed_encoded_data_into_DCT(encoded_bits, dct_blocks):
    data_complete = False
    encoded_bits.pos = 0
    encoded_data_len = bitstring.pack('uint:32', len(encoded_bits))
    converted_blocks = []
    for current_dct_block in dct_blocks:
        for i in range(1, len(current_dct_block)):
            curr_coeff = np.int32(current_dct_block[i])
            if (curr_coeff > 1):
                curr_coeff = np.uint8(current_dct_block[i])
                if (encoded_bits.pos == (len(encoded_bits) - 1)):
                  data_complete = True
                  break
                pack_coeff = bitstring.pack('uint:8', curr_coeff)
        
                if (encoded_data_len.pos <= len(encoded_data_len) - 1):
                  pack_coeff[-1] = encoded_data_len.read(1)
                else:
                  pack_coeff[-1] = encoded_bits.read(1)
                current_dct_block[i] = np.float32(pack_coeff.read('uint:8'))
        converted_blocks.append(current_dct_block)

    if not(data_complete): raise ValueError("Data couldn't be fully embedded into the cover image!")

    return converted_blocks

def zig_zag_coding(input):
	h = 0
	v = 0
	vmin = 0
	hmin = 0

	vmax = input.shape[0]
	hmax = input.shape[1]

	i = 0

	output = np.zeros(( vmax * hmax))

	while ((v < vmax) and (h < hmax)):
	
		if ((h + v) % 2) == 0:
			if (v == vmin):
				output[i] = input[v, h]
				if (h == hmax):
					v = v + 1
				else:
					h = h + 1
				i = i + 1
			elif ((h == hmax -1 ) and (v < vmax)):
				output[i] = input[v, h]
				v = v + 1
				i = i + 1
			elif ((v > vmin) and (h < hmax -1 )):
				output[i] = input[v, h]
				v = v - 1
				h = h + 1
				i = i + 1

		else:
			if ((v == vmax -1) and (h <= hmax -1)):
				output[i] = input[v, h]
				h = h + 1
				i = i + 1
			elif (h == hmin):
				output[i] = input[v, h]
				if (v == vmax -1):
					h = h + 1
				else:
					v = v + 1
				i = i + 1
			elif ((v < vmax -1) and (h > hmin)):
				output[i] = input[v, h]
				v = v + 1
				h = h - 1
				i = i + 1
	
		if ((v == vmax-1) and (h == hmax-1)):
			output[i] = input[v, h]
			break

	return output

def inverse_zig_zag_coding(input, vmax, hmax):
	h = 0
	v = 0

	vmin = 0
	hmin = 0

	output = np.zeros((vmax, hmax))

	i = 0

	while ((v < vmax) and (h < hmax)):

		if ((h + v) % 2) == 0:
			if (v == vmin):
				output[v, h] = input[i]
				if (h == hmax):
					v = v + 1
				else:
					h = h + 1
				i = i + 1
			elif ((h == hmax -1 ) and (v < vmax)):
				output[v, h] = input[i]
				v = v + 1
				i = i + 1
			elif ((v > vmin) and (h < hmax -1 )):
				output[v, h] = input[i]
				v = v - 1
				h = h + 1
				i = i + 1

		else:
			if ((v == vmax -1) and (h <= hmax -1)):
				output[v, h] = input[i]
				h = h + 1
				i = i + 1
			elif (h == hmin):
				output[v, h] = input[i]
				if (v == vmax -1):
					h = h + 1
				else:
					v = v + 1
				i = i + 1
			elif((v < vmax -1) and (h > hmin)):
				output[v, h] = input[i]
				v = v + 1
				h = h - 1
				i = i + 1

		if ((v == vmax-1) and (h == hmax-1)):
			output[v, h] = input[i]
			break

	return output

def extract_encoded_data_from_DCT(dct_blocks):
    extracted_data = ""
    for current_dct_block in dct_blocks:
        for i in range(1, len(current_dct_block)):
            curr_coeff = np.int32(current_dct_block[i])
            if (curr_coeff > 1):
                extracted_data += bitstring.pack('uint:1', np.uint8(current_dct_block[i]) & 0x01)
    return extracted_data

def encoding(COVER_IMAGE_FILEPATH, SECRET_MESSAGE_STRING):

    NUM_CHANNELS = 3

    raw_cover_image = cv2.imread(COVER_IMAGE_FILEPATH, flags=cv2.IMREAD_COLOR)
    height, width   = raw_cover_image.shape[:2]

    # Force Image Dimensions to be 8x8 compliant
    while(height % 8): height += 1
    while(width  % 8): width  += 1
    valid_dim = (width, height)
    padded_image    = cv2.resize(raw_cover_image, valid_dim)
    cover_image_f32 = np.float32(padded_image)
    cover_image_YCC = YCC_Image(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb))

    stego_image = np.empty_like(cover_image_f32)

    for chan_index in range(NUM_CHANNELS):
        # DCT STAGE
        dct_blocks = [cv2.dct(block) for block in cover_image_YCC.channels[chan_index]]

        # QUANTIZATION STAGE
        dct_quants = [np.around(np.divide(item, JPEG_STD_LUM_QUANT_TABLE50)) for item in dct_blocks]

        # Sort DCT coefficients by frequency
        sorted_coefficients = [zig_zag_coding(block) for block in dct_quants]

        if (chan_index == 0):
            secret_data = ""
            for char in SECRET_MESSAGE_STRING.encode('utf-8'): secret_data += bitstring.pack('uint:8', char)
            embedded_dct_blocks   = embed_encoded_data_into_DCT(secret_data, sorted_coefficients)
            desorted_coefficients = [inverse_zig_zag_coding(block, vmax=8,hmax=8) for block in embedded_dct_blocks]
        else:
            desorted_coefficients = [inverse_zig_zag_coding(block, vmax=8,hmax=8) for block in sorted_coefficients]

        # DEQUANTIZATION STAGE
        dct_dequants = [np.multiply(data, JPEG_STD_LUM_QUANT_TABLE50) for data in desorted_coefficients]

        # Inverse DCT Stage
        idct_blocks = [cv2.idct(block) for block in dct_dequants]

        stego_image[:,:,chan_index] = np.asarray(stitch_8x8_blocks(cover_image_YCC.width, idct_blocks))

    stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_YCR_CB2BGR)

    # Clamp Pixel Values to [0 - 255]
    final_stego_image = np.uint8(np.clip(stego_image_BGR, 0, 255))

    cv2.imwrite("saved.jpg", final_stego_image)

def decode(STEGO_IMAGE_FILEPATH):

    stego_image = cv2.imread(STEGO_IMAGE_FILEPATH, flags=cv2.IMREAD_COLOR)
    stego_image_f32 = np.float32(stego_image)
    stego_image_YCC = YCC_Image(cv2.cvtColor(stego_image_f32, cv2.COLOR_BGR2YCrCb))

    # DCT STAGE
    dct_blocks = [cv2.dct(block) for block in stego_image_YCC.channels[0]]

    # QUANTIZATION STAGE
    dct_quants = [np.around(np.divide(item, JPEG_STD_LUM_QUANT_TABLE50)) for item in dct_blocks]

    # Sort DCT coefficients by frequency
    sorted_coefficients = [zig_zag_coding(block) for block in dct_quants]

    # DATA EXTRACTION STAGE
    recovered_data = extract_encoded_data_from_DCT(sorted_coefficients)

    data_len = int(recovered_data.read('uint:32') / 8)

    extracted_data = bytes()
    for _ in range(data_len): extracted_data += struct.pack('>B', recovered_data.read('uint:8'))

    try:
        decoded_text = extracted_data.decode('utf-8')
    except UnicodeDecodeError as e:
    # Handle the error
        decoded_text = extracted_data.decode('utf-8', errors='replace')
    # return extracted_data.decode('utf-8')
    return decoded_text


#########################Telagram part#########################

bot = telebot.TeleBot('6187951636:AAE4iu78entPugPqx7eABsKG4d8LL7RgZHA')

# formatted_data = json.dumps(data, indent=4)

@bot.message_handler(commands=['start'])
def send_welcome(message: telebot.types.Message):
    text = "Hello, this bot was created to see how Image Steganography works in practice.\nType /encode to hide the secret image.\nType /decode to extract the secret image"
    bot.reply_to(message, text)


@bot.message_handler(commands=['encode'])
def encode_handler(message: telebot.types.Message):
    text = "Great! Please send me the cover image."
    sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
    bot.register_next_step_handler(sent_msg, cover_image_handler)

def cover_image_handler(message: telebot.types.Message):
    if message.photo:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        cover_image = bot.download_file(file_info.file_path)
        # Image.SAVE('part.jpg', cover_image)
        with open(f"bro_{message.chat.id}.jpg", "wb") as f:
              f.write(cover_image)
        text = "Great! Now please send me the secret text."
        sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
        bot.register_next_step_handler(sent_msg, secret_text_handler, f"bro_{message.chat.id}.jpg")
    # elif message.document:
    #     file_id = message.document.file_id
    #     file_info = bot.get_file(file_id)
    #     file_extension = file_info.file_path.split('.')[-1].lower()
    #     if file_extension == 'jpg' or file_extension == 'jpeg' or file_extension == 'heic':
    #         cover_image = bot.download_file(file_info.file_path)
    #         text = "Great! Now please send me the secret text."
    #         sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
    #         bot.register_next_step_handler(sent_msg, secret_text_handler, cover_image)
    #     else:
    #         text = "The document is not photo file. Please send me a JPEG/HEIC file as the cover image!"
    #         sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
    #         bot.register_next_step_handler(sent_msg,secret_text_handler)
    else:
        text = "It is not an image. Please send me the cover image!"
        sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
        bot.register_next_step_handler(sent_msg,cover_image_handler)


def secret_text_handler(message: telebot.types.Message, cover_image :str = None):
    if message.text:
        secret_text = message.text
        text = f"Your secret text is: '{secret_text}'"
        # bot.send_document(message.chat.id, cover_image, caption="your cover image")
        # sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
        encoding(cover_image, secret_text)
        with open(r"D:\2year\2semester\LA\project\saved.jpg", "rb") as photo:
        # with open(r"D:\2year\2semester\LA\project\завантаження.jpeg", 'rb') as photo:
            bot.send_photo(message.chat.id, photo, caption="Your stego-image")
    else:
        text = "It is not an text! Please send me the secret text!"
        sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
        bot.register_next_step_handler(sent_msg, secret_text_handler, cover_image)

def part_solver(cover_image, secret_text):
    bot.send_document(681278574, cover_image)
    bot.send_message(681278574, secret_text, parse_mode="Markdown")


@bot.message_handler(commands=['decode'])
def decode_handler(message: telebot.types.Message):
    text = "Great! Please send me the stego-image."
    sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
    bot.register_next_step_handler(sent_msg, stego_image_handler)

def stego_image_handler(message: telebot.types.Message):
    if message.photo:
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        file_path = file_info.file_path
        stego_image = bot.download_file(file_info.file_path)
        # file_extension = file_path.split('.')[-1].lower()
        # if file_extension == 'jpg' or file_extension == 'jpeg' or file_extension == 'heic':
            # stego_image = bot.download_file("")
        with open(f"bro_stego_{message.chat.id}.jpg", "wb") as f:
              f.write(stego_image)
        secret_text = decode(f"bro_stego_{message.chat.id}.jpg")
        bot.send_message(message.chat.id, "Your secret text: " + secret_text, parse_mode="Markdown")
            # with open(r"D:\2year\2semester\LA\project\завантаження.jpeg", 'rb') as photo:
            #     bot.send_document(message.chat.id, photo, caption="Your secret image:")
    elif message.document:
        file_id = message.document.file_id
        file_info = bot.get_file(file_id)
        file_path = file_info.file_path
        stego_image = bot.download_file(file_info.file_path)
        with open(f"bro_stego_{message.chat.id}.jpg", "wb") as f:
            f.write(stego_image)
        secret_text = decode(f"bro_stego_{message.chat.id}.jpg")
        bot.send_message(message.chat.id, "Your secret text: " + secret_text, parse_mode="Markdown")       
    else:
        text = "This is not photo. Please send me a photo as the stego-image!!"
        sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
        bot.register_next_step_handler(sent_msg, stego_image_handler)
    # else:
    #     text = "It is not photo file. Please send me a JPEG/HEIC file as the stego-image!"
    #     sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
    #     bot.register_next_step_handler(sent_msg, stego_image_handler)
    

def part2_solver(stego_image):
    bot.send_document(681278574, stego_image)


@bot.message_handler(func=lambda m: True)
def echo_all(message: telebot.types.Message):
    text = "I don't really understand you. There is only two functions: \nType /encode to hide the secret image.\nType /decode to extract the secret image"
    bot.reply_to(message, text)

bot.infinity_polling()