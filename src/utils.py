import torch

class Utils:

    @staticmethod
    def evaluate(model, loader, criterion, device):
        model.eval()  # Set model to evaluation mode
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_loss = test_loss / len(loader)
        accuracy = 100 * correct / total
        
        print(f'Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%')
        return avg_loss, accuracy
    
    @staticmethod
    def predict_encoding(text, model, dataset, device, max_len=32):
        model.eval()
        
        chars = list(str(text))
        
        idx_seq = [dataset.vocab.get(char, 1) for char in chars]
        
        if len(idx_seq) < max_len:
            idx_seq += [0] * (max_len - len(idx_seq))
        else:
            idx_seq = idx_seq[:max_len]

        input_tensor = torch.tensor(idx_seq).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            
        inv_label_map = {v: k for k, v in dataset.label_map.items()}
        prediction = inv_label_map[predicted_idx.item()]
        
        return prediction