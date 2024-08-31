import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pretty_midi
# import random
import argparse
from sklearn.model_selection import train_test_split
# import logging
from tqdm import tqdm

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
class MIDIDataset(Dataset):
    def __init__(self, midifiles, sequence_length=64):
        self.sequence_length = sequence_length
        self.data = []
        self.max_step = 0
        self.max_duration = 0
        max_file_num = 400

        filenames = glob.glob(midifiles)
        if (len(filenames) < max_file_num):
            max_file_num = len(filenames)
        print(f"Find {len(filenames)} files. Select {max_file_num} for training...")

        # 处理所有MIDI文件
        print("loading midi files...")
        for f in tqdm(filenames[:max_file_num]):
            self.process_midi(f)

        # 数据归一化
        self.normalize_data()

    def process_midi(self, filepath):
        midi_data = pretty_midi.PrettyMIDI(filepath)

        # 选择主旋律轨道（这里简单地选择第一个乐器轨道）
        instrument = midi_data.instruments[0]

        notes = []
        prev_start = 0
        eps = 1e-8  # 一个很小的正数
        self.max_step = max(eps, self.max_step)
        self.max_duration = max(eps, self.max_duration)

        for note in instrument.notes:
            pitch = note.pitch
            start = note.start
            end = note.end
            step = start - prev_start
            duration = end - start
            step = max(0, step)
            duration = max(0, duration)

            self.max_step = max(self.max_step, step)
            self.max_duration = max(self.max_duration, duration)

            notes.append((pitch, step, duration))
            prev_start = start

        # 创建滑动窗口序列
        for i in range(0, len(notes) - self.sequence_length):
            self.data.append(notes[i:i + self.sequence_length])

    def normalize_data(self):
        print("normalize note data...")
        for i, sequence in enumerate(tqdm(self.data)):
            normalized_sequence = []
            for pitch, step, duration in sequence:
                norm_pitch = pitch / 127.0
                norm_step = np.log1p(step) / np.log1p(self.max_step)
                norm_duration = np.log1p(duration) / np.log1p(self.max_duration)
                normalized_sequence.append((norm_pitch, norm_step, norm_duration))
            self.data[i] = normalized_sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        input_seq = torch.tensor(sequence[:-1], dtype=torch.float32)
        target_seq = torch.tensor(sequence[1:], dtype=torch.float32)
        return input_seq, target_seq


class MusicLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=256, num_layers=2, output_size=3):
        super(MusicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out)
        return out

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    print("training...")
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        # print(f"Data shape: {data.shape}")
        # print(f"Data min: {data.min()}, max: {data.max()}, mean: {data.mean()}")
        # print(f"Target shape: {target.shape}")
        # print(f"Target min: {target.min()}, max: {target.max()}, mean: {target.mean()}")

        optimizer.zero_grad()
        output = model(data)

        # print(f"Output shape: {output.shape}")
        # print(f"Output min: {output.min()}, max: {output.max()}, mean: {output.mean()}")

        loss = criterion(output.view(-1, 3), target.view(-1, 3))
        loss.backward()

        # if torch.isnan(loss):
        #     print("NaN loss detected!")
        #     print(f"Output: {output}")
        #     print(f"Target: {target}")
        #     break

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    #print(f'average loss: {total_loss / len(train_loader)}')
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        print("validate...")
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.view(-1, 3), target.view(-1, 3))
            total_loss += loss.item()
    return total_loss / len(val_loader)


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device='cuda'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= 10:
                print("Early stopping")
                break


def generate_music(model, seed_sequence, num_notes=64, temperature=1.0):
    model.eval()
    with torch.no_grad():
        current_sequence = seed_sequence.unsqueeze(0)
        generated_sequence = []

        for _ in range(num_notes):
            output = model(current_sequence)
            #print(f'output: {output}')
            last_output = output[:, -1, :] / temperature
            #print(f'last output: {last_output}')

            # 应用softmax到pitch预测
            # pitch_probs = torch.softmax(last_output[:, 0] * 127, dim=-1)
            # pitch = torch.multinomial(pitch_probs, 1).item() / 127.0
            pitch = torch.clamp(last_output[:, 0], 0, 1).item()
            #print(f'pitch: {pitch}')

            # 对step和duration使用加性高斯噪声
            step = torch.clamp(last_output[:, 1] + torch.randn_like(last_output[:, 1]) * 0.1, 0, 1).item()
            duration = torch.clamp(last_output[:, 2] + torch.randn_like(last_output[:, 2]) * 0.1, 0, 1).item()
            #step = last_output[:, 1].item()
            #duration = last_output[:, 2].item()

            generated_note = torch.tensor([[pitch, step, duration]], dtype=torch.float32)
            generated_sequence.append(generated_note)

            current_sequence = torch.cat([current_sequence[:, 1:, :], generated_note.unsqueeze(0)], dim=1)

        #print(generated_sequence)
    return torch.cat(generated_sequence, dim=0)

def sequence_to_midi(sequence, output_file):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Piano

    current_time = 0
    for note in sequence:
        pitch = int(note[0] * 127)
        step = np.exp(note[1] * np.log(1 + 4)) - 1  # Reverse log normalization, assuming max_step was 4
        duration = np.exp(note[2] * np.log(1 + 4)) - 1  # Reverse log normalization, assuming max_duration was 4

        start = current_time
        end = start + duration
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
        instrument.notes.append(note)
        current_time += step

    midi.instruments.append(instrument)
    midi.write(output_file)
    print(f'Generated MIDI file saved as {output_file}')

def main(args):
    # 数据加载
    dataset = MIDIDataset(args.midi_files, sequence_length=args.sequence_length)
    print('spliting dataset...')
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    print('initialize data loader...')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicLSTM(input_size=3, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)

    # 训练模型
    if not args.generate_only:
        train_model(model, train_loader, val_loader, num_epochs=args.epochs, learning_rate=args.learning_rate,
                        device=device)
    else:
        model.load_state_dict(torch.load('best_model.pth'))
    #model.load_state_dict(torch.load('best_model.pth'))

    # 生成音乐
    seed_sequence = next(iter(val_loader))[0][0].to(device)
    # print(seed_sequence)
    generated_sequence = generate_music(model, seed_sequence, num_notes=args.num_notes,
                                            temperature=args.temperature)

    # 将生成的序列转换为MIDI文件
    sequence_to_midi(generated_sequence.cpu().numpy(), args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM-based Music Generation")
    #parser.add_argument("--midi_files", type=str, default="/Users/MaxGYX/code/MidiGen/midifiles", help="Path to the folder containing MIDI files")
    parser.add_argument("--midi_files", type=str, default="maestro-v2.0.0/*/*.midi", help="Path to the folder containing MIDI files")
    parser.add_argument("--sequence_length", type=int, default=64, help="Length of input sequences")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of the LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_notes", type=int, default=128, help="Number of notes to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for music generation")
    parser.add_argument("--output_file", type=str, default="generated_music.mid", help="Output MIDI file name")
    parser.add_argument("--generate_only", action="store_true",
                            help="Skip training and generate music using saved model")

    args = parser.parse_args()
    main(args)

