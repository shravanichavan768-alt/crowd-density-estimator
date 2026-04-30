import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from model import CSRNet

def process_video(video_path, model, device, output_path='output_video.mp4'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_counts = []
    frame_num    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % 3 != 0:  # process every 3rd frame for speed
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(rgb)

        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)

        density = output.squeeze().cpu().numpy()
        density = cv2.resize(density, (w, h))
        density = np.maximum(density, 0)
        count   = int(density.sum())
        frame_counts.append(count)

        # Heatmap overlay
        density_norm  = density / (density.max() + 1e-8)
        heatmap       = cv2.applyColorMap(
            (density_norm * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # Add count text
        cv2.putText(overlay, f'Count: {count}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        # Add risk level
        density_per_m2 = count / 100
        if density_per_m2 > 7:
            risk, color = 'CRITICAL', (0, 0, 255)
        elif density_per_m2 > 4:
            risk, color = 'DANGER', (0, 140, 255)
        elif density_per_m2 > 2:
            risk, color = 'WARNING', (0, 255, 255)
        else:
            risk, color = 'SAFE', (0, 255, 0)

        cv2.putText(overlay, risk, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        out.write(overlay)

    cap.release()
    out.release()

    return frame_counts