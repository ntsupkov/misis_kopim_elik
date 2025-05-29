import pandas as pd
import numpy as np
import os
from datetime import datetime


def extract_excel_data(file_path):
    excel = pd.ExcelFile(file_path)
    all_data = []

    for sheet_name in excel.sheet_names:
        try:
            sheet_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            for i in range(min(5, sheet_df.shape[0])):
                row_values = [str(val) if pd.notna(val) else "NA" for val in sheet_df.iloc[i, :12]]
            date_row_idx = 0 
            site_row_idx = 1
            shift_row_idx = 2

            indicators = {
                'Переработка_руды_ВМТ': 3,
                'Влажность': 4,
                'Переработка_руды_СМТ': 5,
                'Cu': 7,
                'Zn': 8,
                'S': 9,
                'Гранулометрия': 10
            }

            site1_cols = 0
            site2_cols = 0

            date_data = {}

            date_columns = {}

            for col in range(sheet_df.shape[1]):
                if pd.notna(sheet_df.iloc[date_row_idx, col]):
                    cell_value = sheet_df.iloc[date_row_idx, col]
                    date_value = None

                    if isinstance(cell_value, datetime):
                        date_value = cell_value.date()
                    elif isinstance(cell_value, str) and '.' in cell_value:
                        try:
                            date_value = datetime.strptime(cell_value, '%d.%m.%Y').date()
                        except:
                            pass

                    if date_value:
                        date_columns[col] = date_value
                        if date_value not in date_data:
                            date_data[date_value] = {'Дата': date_value}


            for col in range(sheet_df.shape[1]):
                if col == 0:
                    continue

                site_value = None
                if pd.notna(sheet_df.iloc[site_row_idx, col]):
                    site_text = str(sheet_df.iloc[site_row_idx, col]).strip()
                    if "Участок 1" in site_text:
                        site_value = 1
                    elif "Участок 2" in site_text:
                        site_value = 2

                if site_value is None:
                    continue

                shift_value = None
                if pd.notna(sheet_df.iloc[shift_row_idx, col]):
                    shift_text = str(sheet_df.iloc[shift_row_idx, col])
                    if "1 смена" in shift_text:
                        shift_value = 1
                    elif "2 смена" in shift_text:
                        shift_value = 2

                if shift_value != 1:
                    continue

                date_col = None
                date_value = None

                for c in range(col, -1, -1):
                    if c in date_columns:
                        date_col = c
                        date_value = date_columns[c]
                        break

                if date_value is None:
                    continue

                if site_value == 1:
                    site1_cols += 1
                elif site_value == 2:
                    site2_cols += 1

                for indicator_name, row_idx in indicators.items():
                    if row_idx < sheet_df.shape[0] and pd.notna(sheet_df.iloc[row_idx, col]):
                        column_name = f"{indicator_name}_{site_value}"
                        date_data[date_value][column_name] = sheet_df.iloc[row_idx, col]

            for date_value, row_data in date_data.items():
                if len(row_data) > 1:
                    all_data.append(row_data)

        except Exception as e:
            print(f"Ошибка при обработке листа {sheet_name}: {e}")

    if all_data:
        result_df = pd.DataFrame(all_data)

        site1_cols = [col for col in result_df.columns if col.endswith('_1')]
        site2_cols = [col for col in result_df.columns if col.endswith('_2')]

        if not site1_cols:
            print("ВНИМАНИЕ! Нет данных для Участка 1!")
        if not site2_cols:
            print("ВНИМАНИЕ! Нет данных для Участка 2!")

        for col in sorted(result_df.columns):
            print(f"  - {col}")

        return result_df
    else:
        return pd.DataFrame()


def process_csv_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

        separators = {';': first_line.count(';'),
                      ',': first_line.count(','),
                      '\t': first_line.count('\t')}
        separator = max(separators, key=separators.get)

        df = pd.read_csv(file_path, sep=separator, encoding='utf-8')
        time_column = None
        for col in df.columns:
            if 'время' in col.lower() or 'time' in col.lower() or 'дата' in col.lower():
                time_column = col
                break

        if time_column is None:
            time_column = df.columns[0]

        try:
            df[time_column] = pd.to_datetime(df[time_column])
        except Exception as e:
            print(f"Ошибка при преобразовании времени: {e}")
            formats = ['%Y-%m-%d %H:%M:%S', '%d.%m.%Y %H:%M:%S', '%d.%m.%Y']
            for fmt in formats:
                try:
                    df[time_column] = pd.to_datetime(df[time_column], format=fmt)
                    break
                except:
                    continue
        df['Дата'] = df[time_column].dt.date
        return df, time_column

    except Exception as e:
        print(f"Ошибка при обработке CSV: {e}")
        return None, None


def merge_data(csv_df, excel_df, time_column):
    if csv_df is None or excel_df.empty:
        return None

    csv_df['Дата'] = pd.to_datetime(csv_df['Дата'])
    excel_df['Дата'] = pd.to_datetime(excel_df['Дата'])

    result_df = pd.merge(
        csv_df,
        excel_df,
        on='Дата',
        how='left'
    )

    site1_cols = [col for col in result_df.columns if col.endswith('_1')]
    site2_cols = [col for col in result_df.columns if col.endswith('_2')]

    null_counts = result_df[site1_cols + site2_cols].isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({count / len(result_df) * 100:.1f}%)")

    return result_df


def main():
    excel_path = 'Анализы.xlsx'
    csv_path = 'granulometry_v2.csv'

    excel_df = extract_excel_data(excel_path)

    if excel_df.empty:
        print("\nНевозможно продолжить без данных из Excel")
        return

    csv_df, time_column = process_csv_file(csv_path)

    if csv_df is None:
        print("\nНевозможно продолжить без данных из CSV")
        return

    result_df = merge_data(csv_df, excel_df, time_column)

    if result_df is None:
        print("Не удалось объединить данные")
        return

    output_path = "merged_dataset.csv"
    result_df.to_csv(output_path, index=False, encoding='utf-8')

if __name__ == "__main__":
    main()