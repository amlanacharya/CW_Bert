import streamlit as st
import pandas as pd
from BSA import process_bank_statement, INCOME_LABELS, EXPENSE_LABELS, LABEL_KEYWORDS
from sql_loader import SQLLoader
import tempfile
import os

st.title("Bank Statement Analyzer")

st.write("""
Upload your bank statement CSV file to get a detailed analysis of your income and expenses.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name
    
    try:
        income_df, expenses_df, full_df, labeler = process_bank_statement(tmp_filepath)
        
        if income_df is not None:
            st.success("Analysis completed successfully!")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Income Details", "Expense Details", "Database", "Settings"])
            
            with tab1:
                st.header("Financial Summary")
                total_income = income_df['Amount'].sum()
                total_expenses = expenses_df['Amount'].sum()
                net_amount = total_income - total_expenses
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Income", f"₹{total_income:,.2f}")
                with col2:
                    st.metric("Total Expenses", f"₹{total_expenses:,.2f}")
                with col3:
                    st.metric("Net Profit/Loss", f"₹{net_amount:,.2f}")
                
                st.header("Monthly Breakdown")
                monthly_data = pd.DataFrame(full_df.groupby(full_df['Date'].dt.strftime('%B %Y')).agg({
                    'Amount': lambda x: (x * (full_df.loc[x.index, 'Dr / Cr'] == 'CR')).sum() - 
                                      (x * (full_df.loc[x.index, 'Dr / Cr'] == 'DR')).sum()
                }))
                st.bar_chart(monthly_data)
            
            with tab2:
                st.header("Income Transactions")
                edited_income_df = income_df.copy()
                
                # Add bulk labeling options
                st.subheader("Automatic Labeling Options")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Auto-Label All Income"):
                        edited_income_df = income_df.copy()  # Reset to original
                        # Auto-labeling is already done in process_bank_statement
                
                # Show transactions with suggested labels
                st.subheader("Review and Edit Labels")
                for idx, row in edited_income_df.iterrows():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 0.5])
                    with col1:
                        st.text(f"₹{row['Amount']:,.2f} - {row['Description']}")
                    with col2:
                        suggested_label = row['Label']
                        confidence = row['Confidence']
                        edited_income_df.at[idx, 'Label'] = st.selectbox(
                            f"Label (Confidence: {confidence:.0f}%)",
                            options=INCOME_LABELS,
                            key=f"income_label_{idx}",
                            index=INCOME_LABELS.index(suggested_label) if suggested_label in INCOME_LABELS else 0
                        )
                    with col3:
                        edited_income_df.at[idx, 'Sublabel'] = st.text_input(
                            "Sublabel (optional)",
                            value=row.get('Sublabel', ''),
                            key=f"income_sublabel_{idx}"
                        )
                    with col4:
                        confidence_color = "green" if confidence >= 85 else "orange" if confidence >= 70 else "red"
                        st.markdown(f"<span style='color:{confidence_color}'>{confidence:.0f}%</span>", unsafe_allow_html=True)
                
                # Download buttons for both versions
                col1, col2 = st.columns(2)
                with col1:
                    csv_income = income_df.to_csv(index=False)
                    st.download_button(
                        label="Download Unclean Income Data",
                        data=csv_income,
                        file_name="uc_income.csv",
                        mime="text/csv"
                    )
                with col2:
                    csv_income_labeled = edited_income_df.to_csv(index=False)
                    st.download_button(
                        label="Download Labeled Income Data",
                        data=csv_income_labeled,
                        file_name="labeled_income.csv",
                        mime="text/csv"
                    )
            
            with tab3:
                st.header("Expense Transactions")
                edited_expenses_df = expenses_df.copy()
                
                # Add bulk labeling options for expenses
                st.subheader("Automatic Labeling Options")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Auto-Label All Expenses"):
                        edited_expenses_df = expenses_df.copy()  # Reset to original
                        # Auto-labeling is already done in process_bank_statement
                
                # Show transactions with suggested labels
                st.subheader("Review and Edit Labels")
                for idx, row in edited_expenses_df.iterrows():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 0.5])
                    with col1:
                        st.text(f"₹{row['Amount']:,.2f} - {row['Description']}")
                    with col2:
                        suggested_label = row['Label']
                        confidence = row['Confidence']
                        edited_expenses_df.at[idx, 'Label'] = st.selectbox(
                            f"Label (Confidence: {confidence:.0f}%)",
                            options=EXPENSE_LABELS,
                            key=f"expense_label_{idx}",
                            index=EXPENSE_LABELS.index(suggested_label) if suggested_label in EXPENSE_LABELS else 0
                        )
                    with col3:
                        edited_expenses_df.at[idx, 'Sublabel'] = st.text_input(
                            "Sublabel (optional)",
                            value=row.get('Sublabel', ''),
                            key=f"expense_sublabel_{idx}"
                        )
                    with col4:
                        confidence_color = "green" if confidence >= 85 else "orange" if confidence >= 70 else "red"
                        st.markdown(f"<span style='color:{confidence_color}'>{confidence:.0f}%</span>", unsafe_allow_html=True)
                
                # Download buttons for both versions
                col1, col2 = st.columns(2)
                with col1:
                    csv_expenses = expenses_df.to_csv(index=False)
                    st.download_button(
                        label="Download Unclean Expense Data",
                        data=csv_expenses,
                        file_name="uc_expenses.csv",
                        mime="text/csv"
                    )
                with col2:
                    csv_expenses_labeled = edited_expenses_df.to_csv(index=False)
                    st.download_button(
                        label="Download Labeled Expense Data",
                        data=csv_expenses_labeled,
                        file_name="labeled_expenses.csv",
                        mime="text/csv"
                    )
            
            with tab4:
                st.header("Database Operations")
                if st.button("Export to SQL Database"):
                    loader = SQLLoader()
                    if loader.load_dataframes_to_sql(income_df, expenses_df, full_df):
                        st.success("Successfully exported to SQL database!")
                        
                        info = loader.get_database_info()
                        if info:
                            st.subheader("Database Information")
                            for table, details in info.items():
                                st.write(f"\n**Table**: {table}")
                                st.write(f"Rows: {details['rows']}")
                                st.write(f"Size: {details['size']}")
                                
                if st.button("Delete Database"):
                    loader = SQLLoader()
                    if loader.delete_database():
                        st.success("Database deleted successfully!")
                    else:
                        st.error("Error deleting database")
            
            with tab5:
                st.header("Transaction Classification Settings")
                
                # Keyword Rules Management
                st.subheader("Keyword Rules")
                
                # Select category to edit
                rule_type = st.selectbox("Select Category Type", ["Income", "Expense"])
                labels = INCOME_LABELS if rule_type == "Income" else EXPENSE_LABELS
                selected_label = st.selectbox("Select Label", labels)
                
                # Show existing keywords
                current_keywords = LABEL_KEYWORDS.get(selected_label, [])
                st.write("Current Keywords:", ", ".join(current_keywords))
                
                # Add new keywords
                new_keyword = st.text_input("Add New Keyword")
                if st.button("Add Keyword") and new_keyword:
                    if new_keyword not in current_keywords:
                        LABEL_KEYWORDS[selected_label].append(new_keyword)
                        st.success(f"Added '{new_keyword}' to {selected_label}")
                
                # Remove keywords
                if current_keywords:
                    keyword_to_remove = st.selectbox("Select Keyword to Remove", current_keywords)
                    if st.button("Remove Keyword"):
                        LABEL_KEYWORDS[selected_label].remove(keyword_to_remove)
                        st.success(f"Removed '{keyword_to_remove}' from {selected_label}")
                
                # Bulk Transaction Processing
                st.subheader("Bulk Transaction Processing")
                
                # Select similar transactions
                pattern = st.text_input("Enter text pattern to find similar transactions")
                if pattern:
                    similar_income = income_df[income_df['Description'].str.contains(pattern, case=False, na=False)]
                    similar_expenses = expenses_df[expenses_df['Description'].str.contains(pattern, case=False, na=False)]
                    
                    if len(similar_income) > 0:
                        st.write(f"Found {len(similar_income)} matching income transactions")
                        if st.button("Label Similar Income"):
                            selected_label = st.selectbox("Choose label for all matching income", INCOME_LABELS)
                            sublabel = st.text_input("Optional sublabel for all matching income")
                            if st.button("Apply Income Labels"):
                                income_df.loc[similar_income.index, 'Label'] = selected_label
                                if sublabel:
                                    income_df.loc[similar_income.index, 'Sublabel'] = sublabel
                                st.success("Labels applied successfully!")
                    
                    if len(similar_expenses) > 0:
                        st.write(f"Found {len(similar_expenses)} matching expense transactions")
                        if st.button("Label Similar Expenses"):
                            selected_label = st.selectbox("Choose label for all matching expenses", EXPENSE_LABELS)
                            sublabel = st.text_input("Optional sublabel for all matching expenses")
                            if st.button("Apply Expense Labels"):
                                expenses_df.loc[similar_expenses.index, 'Label'] = selected_label
                                if sublabel:
                                    expenses_df.loc[similar_expenses.index, 'Sublabel'] = sublabel
                                st.success("Labels applied successfully!")
        
                st.subheader("Fuzzy Matching Settings")
                new_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=50,
                    max_value=100,
                    value=85,
                    help="Minimum similarity score required for fuzzy matching (higher = stricter matching)"
                )
                
                if st.button("Update Similarity Threshold"):
                    labeler.similarity_threshold = new_threshold
                    st.success(f"Updated similarity threshold to {new_threshold}")
                
                st.subheader("BERT Model Settings")
                use_bert = st.checkbox("Use BERT for Classification", value=True)
                
                if st.button("Train BERT Model"):
                    # Collect training data from labeled transactions
                    texts = []
                    labels = []
                    
                    # Add income transactions
                    for _, row in edited_income_df.iterrows():
                        if row['Label']:  # Only use labeled transactions
                            texts.append(row['Description'])
                            labels.append(row['Label'])
                    
                    # Add expense transactions
                    for _, row in edited_expenses_df.iterrows():
                        if row['Label']:  # Only use labeled transactions
                            texts.append(row['Description'])
                            labels.append(row['Label'])
                    
                    if texts and labels:
                        with st.spinner("Training BERT model..."):
                            success = labeler.bert_labeler.train(texts, labels)
                            if success:
                                st.success("BERT model trained successfully!")
                            else:
                                st.error("Error training BERT model")
                    else:
                        st.warning("No labeled transactions available for training")
                
                # Update labeler settings
                labeler.use_bert = use_bert
                
                st.subheader("Model Management")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Download Trained Model"):
                        filename = labeler.bert_labeler.save_model_pickle()
                        if filename:
                            with open(filename, 'rb') as f:
                                st.download_button(
                                    label="Download Model File",
                                    data=f,
                                    file_name=filename,
                                    mime="application/octet-stream"
                                )
                            os.remove(filename)
                        else:
                            st.error("No trained model available to download")
                
                with col2:
                    uploaded_model = st.file_uploader("Upload Trained Model", type=['pkl'])
                    if uploaded_model:
                        version = uploaded_model.name.split('_')[2].split('.')[0] if '_' in uploaded_model.name else 'unknown'
                        st.info(f"Uploading model version: {version}")
                        
                        with open('temp_model.pkl', 'wb') as f:
                            f.write(uploaded_model.getvalue())
                        if labeler.bert_labeler.load_model_pickle('temp_model.pkl'):
                            st.success(f"Model version {version} loaded successfully!")
                        else:
                            st.error("Error loading model")
                        os.remove('temp_model.pkl')
        
        else:
            st.error("Error processing the file. Please check the file format.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    finally:
        os.unlink(tmp_filepath)

