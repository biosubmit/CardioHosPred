#Define columns to select
def select_columns(combined_data):
    # Select columns to keep
    selected_columns = ['病案号',      # medical_record_number
                        '住院次数',    # hospitalization_count
                        '费别',        # payment_category
                        '性别',        # gender
                        '年龄',        # age
                        '主要手术操作名称',  # main_operation_name
                        '入院时间',    # admission_time
                        '出院科别(首页)', # discharge_department
                        '出院时间',    # discharge_time
                        '出院主要诊断名称1', # discharge_main_diagnosis_name1
                        '病理诊断']    # pathological_diagnosis
    
    # Select columns to keep
    combined_data = combined_data[selected_columns]
    return combined_data
