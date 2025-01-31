import torch

class SpectrumTrainer:
    def __init__(self, model, target_layers, train_data, eval_data, output_dir):
        self.model = model
        self.target_layers = target_layers
        self.train_data = train_data
        self.eval_data = eval_data
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):  # Define num_epochs
            for batch in self.train_data:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        torch.save(self.model.state_dict(), self.output_dir + "/model.pth")

    def evaluate(self):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in self.eval_data:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        accuracy = 100 * total_correct / total_samples
        print(f'Accuracy: {accuracy:.2f}%')
