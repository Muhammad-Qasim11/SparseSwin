import torch
import os
import numpy as np
import pandas as pd

def train(train_loader, swin_type, dataset, total_epochs, model, lf, token_num, # total_epochs & start_epoch added
                optimizer, criterion, device, show_per,
                reg_type=None, reg_lambda=0., validation=None, start_epoch=0): # Default start_epoch to 0
    model.train()
    total_batch = train_loader.__len__()
    train_test_hist = []
    best_test_acc = -99 # This will be reset if resuming. If you need to persist best_test_acc, it should be saved/loaded in checkpoint too.

    # --- Robust directory creation for SavedModel ---
    base_saved_dir = f'./SavedModel/{dataset}'
    os.makedirs(base_saved_dir, exist_ok=True)

    specific_dir = f'{base_saved_dir}/SparseSwin_reg_{reg_type}_lbd_{reg_lambda}_lf_{lf}_{token_num}'
    os.makedirs(specific_dir, exist_ok=True)

    # --- Robust directory creation for TrainingState ---
    training_state_dir = f'./TrainingState/{dataset}'
    os.makedirs(training_state_dir, exist_ok=True)

    print(f"[TRAIN] Total : {total_batch} | type : {swin_type} | Regularization : {reg_type} with lamda : {reg_lambda}")
    for epoch in range(start_epoch, total_epochs): # Loop now starts from start_epoch
        print(f"Epoch {epoch+1}/{total_epochs}") # Correct epoch numbering based on total_epochs
        running_loss, n_correct, n_sample = 0.0, 0.0, 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            if swin_type.lower() == "swin_transformer_tiny" or swin_type.lower() == "swin_transformer_small" or swin_type.lower() == "swin_transformer_base":
                outputs = model(inputs)
            else:
                outputs, attn_weights, _ = model(inputs)

            reg = 0
            if reg_type == 'l1':
                for attn_w in attn_weights:
                    reg += torch.sum(torch.abs(attn_w))
            elif reg_type == 'l2':
                for attn_w in attn_weights:
                    reg += torch.sum(attn_w**2)

            reg = reg_lambda * reg

            loss = criterion(outputs, labels) + reg
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                n_correct_per_batch = torch.sum(torch.argmax(outputs, dim=1) == labels)
                n_correct += n_correct_per_batch
                n_sample += labels.shape[0]
                acc = n_correct / n_sample

            if ((i + 1) % show_per == 0) or ((i + 1) == total_batch):
                print(f'  [{i + 1}/{total_batch}] Loss: {(running_loss / (i + 1)):.4f} Acc : {acc:.4f}')

        print(f'Loss: {(running_loss / total_batch):.4f} Acc : {(n_correct / n_sample):.4f}')

        # Save model if it's the best so far on test accuracy
        test_loss, test_acc = test(validation, swin_type=swin_type, model=model, criterion=criterion, device=device)
        train_loss, train_acc = (running_loss / total_batch), (n_correct / n_sample)

        test_loss, train_loss = round(test_loss, 4), round(train_loss, 4)
        train_test_hist.append([train_loss, round(train_acc.item(), 4), test_loss, round(test_acc.item(), 4)])

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'{specific_dir}/model_{epoch+1}.pt') # Saves best model to SavedModel

    # Save training history to CSV
    train_test_hist = np.array(train_test_hist)
    df = pd.DataFrame()
    df['train_loss'] = train_test_hist[:, 0]
    df['train_acc'] = train_test_hist[:, 1]
    df['test_loss'] = train_test_hist[:, 2]
    df['test_acc'] = train_test_hist[:, 3]
    df.to_csv(f'{specific_dir}/hist.csv', index=None)

    # Save the full training state (model, optimizer, epoch, loss) for resuming
    torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss},
                        f'{training_state_dir}/SparseSwin_{reg_type}_{reg_lambda}_lf_{lf}_{epoch+1}')
    print('Finished Training, saved training state :D')
    print("Train Loss, Train Acc, Test Loss, Test Acc")
    print(train_test_hist)

def test(val_loader, swin_type, model, criterion, device):
    model.eval() # Set model to evaluation mode

    with torch.no_grad(): # Disable gradient calculations for testing
        total_batch = val_loader.__len__()
        print(f"[TEST] Total : {total_batch} | type : {swin_type}")
        running_loss, n_correct, n_sample = 0.0, 0.0, 0.0

        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass
            if swin_type.lower() == "swin_transformer_tiny" or swin_type.lower() == "swin_transformer_small" or swin_type.lower() == "swin_transformer_base":
                outputs = model(inputs)
            else:
                outputs, attn_weights, _ = model(inputs) # Note: _ for the third return value

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            n_correct_per_batch = torch.sum(torch.argmax(outputs, dim=1) == labels)
            n_correct += n_correct_per_batch
            n_sample += labels.shape[0]
            acc = n_correct / n_sample

    print(f'[Model : {swin_type}] Loss: {(running_loss / total_batch):.4f} Acc : {(n_correct / n_sample):.4f}')
    print()
    return (running_loss / total_batch), (n_correct / n_sample)
