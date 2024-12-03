width=416
height=416
stride=2
imgsz = opt.img_size  # Default input size
imgszs = [imgsz - x % 32 for x in range(-3, 4)]  # Multi-scale selection, 7 sizes were selected

for imgsz in imgszs:
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Executive model reasoning
    pred = model(img, augment=opt.augment)[0]
