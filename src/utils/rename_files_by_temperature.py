import os
import re
from datetime import datetime

from src.utils.config_manager import load_config


def parse_timestamp_from_filename(filename):
    """
    Args:
        filename: ("2.50_Slime_slow_2024-10-24_20.12.42.wav")

    Returns:
        datetime object or None if not found
    """
    # Pattern: YYYY-MM-DD_HH.MM.SS
    pattern = r'(\d{4}-\d{2}-\d{2})_(\d{2})\.(\d{2})\.(\d{2})'
    match = re.search(pattern, filename)

    if match:
        date_part = match.group(1)  # YYYY-MM-DD
        hour = match.group(2)
        minute = match.group(3)
        second = match.group(4)

        datetime_str = f"{date_part} {hour}:{minute}:{second}"
        return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

    return None


def read_temperatures_from_file(temperature_file_path):
    """
    Args:
        temperature_file_path

    Returns:
        Floats list of temperatures
    """
    temperatures = []

    with open(temperature_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    temp = float(line.replace(',', '.'))
                    temperatures.append(temp)
                except ValueError:
                    print(f"WARNING Cannot parse temperature: '{line}'")

    return temperatures


def get_audio_files_sorted_by_time(directory):
    """
    Args:
        directory: Path to directory with audio files

    Returns:
        List of tuples: (full_path, filename, timestamp)
    """
    audio_files = []

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.wav'):
                full_path = os.path.join(root, filename)
                timestamp = parse_timestamp_from_filename(filename)

                if timestamp:
                    audio_files.append((full_path, filename, timestamp))
                else:
                    print(f"Can't load timestamp: {filename}")

    audio_files.sort(key=lambda x: x[2])

    return audio_files


def rename_files_with_temperatures(audio_directory, temperature_file_path, dry_run=True):
    """
    Args:
        audio_directory
        temperature_file_path: Path to text file with temperatures (one per line)
        dry_run: If True, only preview changes without renaming files
    """
    print(f"Audio directory: {audio_directory}")
    print(f"Temperatures file: {temperature_file_path}")
    print()

    temperatures = read_temperatures_from_file(temperature_file_path)
    print(f"Read {len(temperatures)} temperatures")
    print(f"Range: {min(temperatures):.2f}°C - {max(temperatures):.2f}°C")
    print()

    audio_files = get_audio_files_sorted_by_time(audio_directory)
    print(f"Found {len(audio_files)} audio files")
    print()

    if len(temperatures) != len(audio_files):
        print(f"WARNING: Temperatures count does not much file count")

        if len(temperatures) < len(audio_files):
            print(f"Missing {len(audio_files) - len(temperatures)} temperatures!")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return

    print("CHANGES:")

    renamed_count = 0
    skipped_count = 0

    for i, (file_path, old_filename, timestamp) in enumerate(audio_files):
        if i >= len(temperatures):
            print(f"Skipped: {old_filename}")
            skipped_count += 1
            continue

        temperature = temperatures[i]

        temp_str = f"{temperature:.2f}"

        new_filename = re.sub(r'^[\d,\.]+', temp_str, old_filename)

        if new_filename == old_filename:
            new_filename = f"{temp_str}_{old_filename}"

        new_file_path = os.path.join(os.path.dirname(file_path), new_filename)

        print(f"[{i+1}/{len(audio_files)}] {timestamp.strftime('%Y-%m-%d %H:%M:%S')} → {temperature:.2f}°C")
        print(f"Old: {old_filename}")
        print(f"New:  {new_filename}")

        if not dry_run:
            try:
                os.rename(file_path, new_file_path)

                processed_old = file_path.replace('.wav', '.processed.wav')
                if os.path.exists(processed_old):
                    processed_new = new_file_path.replace('.wav', '.processed.wav')
                    os.rename(processed_old, processed_new)

                json_old_name = old_filename.replace('.wav', '.json')
                json_new_name = new_filename.replace('.wav', '.json')

                possible_json_dirs = [
                    os.path.join(os.path.dirname(file_path), 'annotations'),
                    os.path.join(os.path.dirname(os.path.dirname(file_path)), 'annotations'),
                ]

                for json_dir in possible_json_dirs:
                    json_old_path = os.path.join(json_dir, json_old_name)
                    if os.path.exists(json_old_path):
                        json_new_path = os.path.join(json_dir, json_new_name)
                        os.rename(json_old_path, json_new_path)
                        print(f"Json changed also: {json_old_name} to {json_new_name}")
                        break

            except Exception as e:
                print(f"ERROR: {e}")

        print()
        renamed_count += 1


if __name__ == "__main__":
    config = load_config("..\\config\\config.yaml")
    audio_directory = config["data"]["data_root"]
    temperature_file_path = config["data"]["temperature_file"]
    rename_files_with_temperatures(audio_directory, temperature_file_path, dry_run=False)
