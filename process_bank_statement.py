import pandas as pd
from transaction_labeler.labeler.labelers import TransactionLabeler
def clean_amount(amount_str):
    if isinstance(amount_str, str):
        return float(amount_str.replace('"', '').replace('=', '').replace(',', ''))
    return amount_str

def process_bank_statement(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        start_row = 0
        end_row = len(lines)
        for i, line in enumerate(lines):
            if 'Sl. No.' in line:
                start_row = i
            if 'Opening balance' in line:
                end_row = i
                break
        
        df = pd.read_csv(file_path, skiprows=start_row, nrows=end_row-start_row)
        df.columns = df.columns.str.strip()
        
        print("\n=== Initial DataFrame ===")
        print("DataFrame shape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nSample data:")
        print(df.head())
        print("\nDataFrame Info:")
        df.info()
        
        df['Amount'] = df['Amount'].apply(clean_amount)
        df['Balance'] = df['Balance'].apply(clean_amount)
        
        # Sort the main dataframe first by Date and then by Sl. No.
        df['Sl. No.'] = pd.to_numeric(df['Sl. No.'], errors='coerce')
        df = df.sort_values(['Date', 'Sl. No.'], ascending=[True, True])
        
        labeler = TransactionLabeler()
        
        # Process income transactions
        income = df[df['Dr / Cr'] == 'CR'].copy()
        income['Label'] = ''
        income['Sublabel'] = ''
        income['Confidence'] = 0  # Add confidence score
        income['Clean_Description'] = income['Description'].apply(labeler.clean_description)
        
        # Apply multiple labeling methods
        def label_income(description):
            label, confidence = labeler.label_transaction(description, 'CR')
            return pd.Series([label, confidence])
        income[['Label', 'Confidence']] = income['Clean_Description'].apply(label_income)

        # Process expense transactions
        expenses = df[df['Dr / Cr'] == 'DR'].copy()
        expenses['Label'] = ''
        expenses['Sublabel'] = ''
        expenses['Confidence'] = 0  # Add confidence score
        expenses['Clean_Description'] = expenses['Description'].apply(labeler.clean_description)
        
        def label_expense(description):
            label, confidence = labeler.label_transaction(description, 'DR')
            return pd.Series([label, confidence])
        expenses[['Label', 'Confidence']] = expenses['Clean_Description'].apply(label_expense)

       
        income.to_csv('uc_income.csv', index=False)
        expenses.to_csv('uc_expenses.csv', index=False)
        
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df['Year_Month'] = df['Date'].dt.strftime('%B %Y')
        monthly_summary = []
        
        for month in sorted(df['Year_Month'].unique(), key=lambda x: pd.to_datetime(x, format='%B %Y')):
            month_data = df[df['Year_Month'] == month]
            month_income = month_data[month_data['Dr / Cr'] == 'CR']['Amount'].sum()
            month_expenses = month_data[month_data['Dr / Cr'] == 'DR']['Amount'].sum()
            net_amount = month_income - month_expenses
            profit_or_loss = "Profit" if net_amount >= 0 else "Loss"
            monthly_summary.append({
                'Month': month,
                'Income': month_income,
                'Expenses': month_expenses,
                'Net': net_amount
            })
        
        print("\n=== Monthly Summary ===")
        for month in monthly_summary:
            print(f"{month['Month']}:")
            print(f"  Income: ₹{month['Income']:,.2f}")
            print(f"  Expenses: ₹{month['Expenses']:,.2f}")
            print(f"  Net: ₹{abs(month['Net']):,.2f} ({'Profit' if month['Net'] >= 0 else 'Loss'})")
            print()

        return income, expenses, df, labeler
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None, None, None, None
