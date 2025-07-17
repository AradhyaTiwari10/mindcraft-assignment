from PIL import Image, ImageDraw, ImageFont
import os

print("[DEBUG] Starting Driving License sample image creation...")

def create_sample_driving_license():
    """Create a sample driving license image for testing."""
    
    # Create a new image with white background
    width, height = 800, 500
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to load fonts, with fallbacks for Unicode support
    try:
        # Try to find a Unicode-compatible font
        font_paths = [
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "/System/Library/Fonts/Helvetica.ttc",  # macOS alternative
        ]
        
        title_font = None
        normal_font = None
        small_font = None
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    title_font = ImageFont.truetype(font_path, 24)
                    normal_font = ImageFont.truetype(font_path, 16)
                    small_font = ImageFont.truetype(font_path, 12)
                    print(f"[DEBUG] Using font: {font_path}")
                    break
                except:
                    continue
        
        # If no system font found, use default
        if title_font is None:
            title_font = ImageFont.load_default()
            normal_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            print("[DEBUG] Using default font")
            
    except Exception as e:
        print(f"[DEBUG] Font loading error: {e}")
        title_font = ImageFont.load_default()
        normal_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw header
    draw.text((50, 30), "Indian Union Driving Licence", fill='blue', font=title_font)
    draw.text((50, 60), "Issued by GOVERNMENT OF RAJASTHAN", fill='black', font=normal_font)
    
    # Draw sample data
    y_position = 120
    
    # License Number
    draw.text((50, y_position), "License Number:", fill='black', font=normal_font)
    draw.text((200, y_position), "RJ14D20240028909", fill='black', font=normal_font)
    y_position += 30
    
    # Issue Date
    draw.text((50, y_position), "Issue Date:", fill='black', font=normal_font)
    draw.text((200, y_position), "06-11-2024", fill='black', font=normal_font)
    y_position += 30
    
    # Validity
    draw.text((50, y_position), "Validity (NT):", fill='black', font=normal_font)
    draw.text((200, y_position), "12-07-2046", fill='black', font=normal_font)
    y_position += 30
    
    # Name
    draw.text((50, y_position), "Name:", fill='black', font=normal_font)
    draw.text((200, y_position), "ARADHYA TIWARI", fill='black', font=normal_font)
    y_position += 30
    
    # Date of Birth
    draw.text((50, y_position), "Date Of Birth:", fill='black', font=normal_font)
    draw.text((200, y_position), "13-07-2006", fill='black', font=normal_font)
    y_position += 30
    
    # Son/Daughter/Wife of
    draw.text((50, y_position), "Son/Daughter/Wife of:", fill='black', font=normal_font)
    draw.text((200, y_position), "GOPESH TIWARI", fill='black', font=normal_font)
    y_position += 30
    
    # Address
    draw.text((50, y_position), "Address:", fill='black', font=normal_font)
    y_position += 25
    draw.text((200, y_position), "94, Luv Kush Nagar 1st Tonk Phatak", fill='black', font=normal_font)
    y_position += 25
    draw.text((200, y_position), "LAL KOTHI Jaipur Rajasthan, 302015", fill='black', font=normal_font)
    y_position += 30
    
    # Holder's Signature
    draw.text((50, y_position), "Holder's Signature:", fill='black', font=normal_font)
    
    # Draw a simple photo placeholder (rectangle)
    photo_x, photo_y = 600, 120
    photo_size = 100
    draw.rectangle([(photo_x, photo_y), (photo_x + photo_size, photo_y + photo_size)], 
                   outline='black', width=2)
    draw.text((photo_x + 10, photo_y + 40), "PHOTO", fill='gray', font=small_font)
    
    # Draw border
    draw.rectangle([(20, 20), (width-20, height-20)], outline='blue', width=2)
    
    # Add some decorative elements
    draw.rectangle([(50, 100), (width-50, 110)], fill='blue')
    
    # Save the image
    os.makedirs('sample_images', exist_ok=True)
    print("[DEBUG] About to save image to sample_images/sample_driving_license.jpg")
    image.save('sample_images/sample_driving_license.jpg', 'JPEG', quality=95)
    print("Sample Driving License created: sample_images/sample_driving_license.jpg")

if __name__ == "__main__":
    create_sample_driving_license() 