from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import shutil

def create_stress_test_samples(output_dir: str = "data/test_samples"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Authentic Real Leaf Samples for Web/Camera Demo Testing
    real_sources = {
        "early_blight": Path("data/raw/tomato_early_blight_0001/initial_view.jpg"),
        "late_blight": Path("data/raw/tomato_late_blight_0001/initial_view.jpg"),
        "septoria": Path("data/raw/tomato_septoria_leaf_spot_0001/initial_view.jpg"),
        "healthy": Path("data/raw/tomato_healthy_0001/initial_view.jpg")
    }

    for name, src in real_sources.items():
        if src.exists():
            shutil.copy(src, out_path / f"sample_real_{name}.jpg")

    # 2. Healthy Leaf for Contradiction Test (Scenario B)
    healthy_src = real_sources["healthy"]
    if healthy_src.exists():
        shutil.copy(healthy_src, out_path / "sample_healthy_contradiction.jpg")
        base_img = Image.open(healthy_src).convert("RGB")
    else:
        base_img = Image.new("RGB", (224, 224), color=(34, 139, 34))
        base_img.save(out_path / "sample_healthy_contradiction.jpg")

    # 3. Blurry Leaf (Scenario C - Reliability Rejection)
    blurry_img = base_img.filter(ImageFilter.GaussianBlur(radius=14))
    blurry_img.save(out_path / "sample_blurry.jpg", quality=90)

    # 4. Underexposed / Dark Leaf (Scenario C - Lighting Rejection)
    dark_enhancer = ImageEnhance.Brightness(base_img)
    dark_img = dark_enhancer.enhance(0.08)
    dark_img.save(out_path / "sample_dark_underexposed.jpg", quality=90)

    # 5. Real Late Blight Underside
    lb_under_src = Path("data/raw/tomato_late_blight_0001/leaf_underside.jpg")
    if lb_under_src.exists():
        shutil.copy(lb_under_src, out_path / "sample_valid_late_blight_underside.jpg")

    print(f"Authentic stress-test and web demonstration samples created in {output_dir}:")
    print(f" 1. sample_real_early_blight.jpg  -> Real Early Blight foliar photo (Alternaria solani)")
    print(f" 2. sample_real_late_blight.jpg   -> Real Late Blight foliar photo (Phytophthora infestans)")
    print(f" 3. sample_real_septoria.jpg      -> Real Septoria Leaf Spot photo (Septoria lycopersici)")
    print(f" 4. sample_real_healthy.jpg       -> Real Healthy tomato foliage")
    print(f" 5. sample_healthy_contradiction.jpg -> Scenario B (Contradiction & Wrong leaf)")
    print(f" 6. sample_blurry.jpg             -> Scenario C (Blur Rejection)")
    print(f" 7. sample_dark_underexposed.jpg  -> Scenario C (Darkness Rejection)")

if __name__ == "__main__":
    create_stress_test_samples()
