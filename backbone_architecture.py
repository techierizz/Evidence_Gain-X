import torch
from src.models.classifier import DiseaseClassifier

def run_demo():
    print("🚀 Initializing EvidenceGain-X Foundation Model...")
    
    classes = ["Early Blight", "Late Blight", "Septoria Leaf Spot", "Healthy"]
    model = DiseaseClassifier(num_classes=len(classes), backbone_name="resnet18", pretrained=False)
    model.eval()
    print(model.backbone)
    print("✅ Model initialized successfully (ResNet-18 Backbone).")
    
    print("\n📸 Simulating an incoming 224x224 crop image...")
    dummy_image = torch.rand(1, 3, 224, 224)

    print("🧠 Running provisional diagnosis...\n")
    with torch.no_grad():
        results = model.predict_provisional(dummy_image)

    print("-" * 40)
    print("📊 DIAGNOSTIC RESULTS:")
    print("-" * 40)
    
    probs = results["probs"][0].tolist()
    for i, disease in enumerate(classes):
        print(f"  {disease}: {probs[i]*100:.2f}%")
    
    print("-" * 40)
    print(f"📉 Calculated Entropy (Uncertainty): {results['entropy'][0].item():.4f}")
    print(f"⚠️ Prediction Margin (Top 1 vs Top 2): {results['margin'][0].item():.4f}")
    print("-" * 40)
    
    print("\n💡 Note: High entropy means the model is uncertain and requires the next module (Counterfactual Planner) to ask for more evidence!")

if __name__ == "__main__":
    run_demo()
