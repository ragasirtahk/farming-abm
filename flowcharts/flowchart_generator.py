from graphviz import Digraph

def create_flowchart():
    dot = Digraph(comment='Farming ABM Flowchart')

    # Define model
    dot.node('Model', 'FarmingModel')
    
    # Initialization of parameters
    dot.node('Init_Params', 'Initialize Parameters:\n- num_farmers\n- num_seasons\n- forecast_accuracy\n- subsidy_level\n- dissemination_modes\n- seed')
    dot.edge('Model', 'Init_Params', label='Initializes')
    
    # Seed setting
    dot.node('Set_Seed', 'Set Seed for Reproducibility')
    dot.edge('Init_Params', 'Set_Seed', label='If seed provided')
    
    # Scaling initial wealth and land
    dot.node('Scale_Wealth', 'Scale Initial Wealth\nBased on total_initial_wealth')
    dot.edge('Set_Seed', 'Scale_Wealth', label='Scale Wealth')
    
    dot.node('Scale_Land', 'Scale Initial Land Sizes\nBased on total_initial_land')
    dot.edge('Scale_Wealth', 'Scale_Land', label='Scale Land Sizes')
    
    # Define agents
    dot.node('Farmer', 'Farmer Agent')
    dot.node('Forecaster', 'Forecaster Agent')

    dot.edge('Model', 'Farmer', label='Initializes Farmers')
    dot.edge('Model', 'Forecaster', label='Initializes Forecaster')

    # Farmer variables
    dot.node('Farmer_Vars', 'Variables:\n- land_size\n- wealth\n- type\n- trust_level\n- crop_type\n- total_yield\n- dissemination_mode')
    dot.edge('Farmer', 'Farmer_Vars')
    
    # Forecaster variables
    dot.node('Forecaster_Vars', 'Variables:\n- accuracy\n- dissemination_mode\n- rainfall_variability')
    dot.edge('Forecaster', 'Forecaster_Vars')

    # Interaction rules
    dot.edge('Forecaster', 'Farmer', label='Provides Forecast')
    dot.edge('Farmer', 'Forecaster', label='Updates Trust')
    
    # Farmer decision-making
    dot.node('Decision', 'Decide Crop')
    dot.edge('Farmer', 'Decision', label='Based on Forecast & Risk Tolerance')
    
    # Yield calculation
    dot.node('Yield', 'Calculate Yield')
    dot.edge('Decision', 'Yield', label='Apply Yield Rules')
    dot.edge('Yield', 'Farmer', label='Update Wealth and Total Yield\n(+ Subsidy Level)')
    
    # Trust update
    dot.node('Trust', 'Update Trust Level')
    dot.edge('Yield', 'Trust', label='Based on Earnings and Forecast Accuracy')
    dot.edge('Trust', 'Farmer', label='Update Trust Level')
    
    # Forecast accuracy crash
    dot.node('Forecast_Crash', 'Forecast Accuracy Crash\n(5% Chance)')
    dot.edge('Forecaster', 'Forecast_Crash', label='Possible Crash')
    dot.edge('Forecast_Crash', 'Trust', label='If Crash Occurs')
    
    # Data collection
    dot.node('Data', 'Data Collection')
    dot.edge('Model', 'Data', label='Collects Data')
    dot.edge('Farmer', 'Data', label='Provides Data')
    dot.edge('Forecaster', 'Data', label='Provides Data')
    
    # Model step
    dot.node('Model_Step', 'Advance Model by One Step')
    dot.edge('Model', 'Model_Step')
    dot.edge('Model_Step', 'Data', label='Collect Data')
    dot.edge('Model_Step', 'Farmer', label='Schedule Step')
    dot.edge('Model_Step', 'Forecaster', label='Schedule Step')
    
    # Export to file
    dot.render('farming_abm_flowchart.gv', view=True)

def create_trust_by_dissemination_mode_flowchart():
    dot = Digraph(comment='Trust by Dissemination Mode')
    
    # Define start
    dot.node('Start', 'Start')
    
    # Determine dissemination modes
    dot.node('DetermineDM', 'Determine Dissemination Modes')
    dot.edge('Start', 'DetermineDM', label='Determine Modes')
    
    # Define dissemination modes
    dot.node('DM1', 'Radio')
    dot.node('DM2', 'Mobile App')
    dot.node('DM3', 'Extension Officer')
    dot.node('DM4', 'Community Leaders')
    
    # Define trust adjustment based on dissemination mode
    dot.edge('DetermineDM', 'DM1', label='Radio')
    dot.edge('DetermineDM', 'DM2', label='Mobile App')
    dot.edge('DetermineDM', 'DM3', label='Extension Officer')
    dot.edge('DetermineDM', 'DM4', label='Community Leaders')
    
    # Adjust Trust
    dot.node('AdjustTrust', 'Adjust Trust Level')
    dot.edge('DM1', 'AdjustTrust', label='Less Personal')
    dot.edge('DM2', 'AdjustTrust', label='Baseline')
    dot.edge('DM3', 'AdjustTrust', label='Personalized Advice')
    dot.edge('DM4', 'AdjustTrust', label='Community Influence')
    
    # Loop through seasons
    dot.node('Loop', 'For Each Season')
    dot.edge('DetermineDM', 'Loop', label='Start Simulation')
    
    # Step 1: Provide Forecast
    dot.node('Forecast', 'Provide Forecast')
    dot.edge('Loop', 'Forecast')
    
    # Step 2: Farmer Decides Crop
    dot.node('DecideCrop', 'Farmer Decides Crop')
    dot.edge('Forecast', 'DecideCrop', label='Based on Forecast & Rules')
    
    # Add decision rules for crop selection
    dot.node('CropRule', 'Determine Crop Type\n(Rainfall & Risk Tolerance)')
    dot.edge('Forecast', 'CropRule', label='Evaluate Conditions')
    dot.edge('CropRule', 'DecideCrop', label='Select Crop')
    
    # Step 3: Calculate Yield with Rules
    dot.node('CalculateYield', 'Calculate Yield')
    dot.edge('DecideCrop', 'CalculateYield', label='Apply Yield Rules')
    
    # Step 4: Update Wealth and Yield
    dot.node('UpdateWealth', 'Update Wealth and Total Yield')
    dot.edge('CalculateYield', 'UpdateWealth')
    
    # Step 5: Update Trust
    dot.node('UpdateTrust', 'Update Trust Level')
    dot.edge('UpdateWealth', 'UpdateTrust')
    
    # Repeat Loop
    dot.edge('UpdateTrust', 'Loop', label='Next Season')
    
    # Define end
    dot.edge('Loop', 'End', style='dotted', label='Simulation End')
    
    # Define end
    dot.node('End', 'End')
    
    dot.render('trust_by_dissemination_mode_flowchart', view=True)

def create_trust_dynamics_over_season_flowchart():
    dot = Digraph(comment='Trust Dynamics Over Seasons')
    
    # Define start
    dot.node('Start', 'Start Season')
    
    # Step 1: Provide Forecast
    dot.node('Forecast', 'Provide Forecast')
    dot.edge('Start', 'Forecast')
    
    # Step 2: Farmer Decides Crop
    dot.node('DecideCrop', 'Farmer Decides Crop')
    dot.edge('Forecast', 'DecideCrop', label='Based on Forecast')
    
    # Step 3: Calculate Yield
    dot.node('CalculateYield', 'Calculate Yield')
    dot.edge('DecideCrop', 'CalculateYield', label='Crop Type')
    
    # Step 4: Update Wealth and Yield
    dot.node('UpdateWealth', 'Update Wealth and Total Yield')
    dot.edge('CalculateYield', 'UpdateWealth')
    
    # Step 5: Update Trust
    dot.node('UpdateTrust', 'Update Trust Level')
    dot.edge('UpdateWealth', 'UpdateTrust')
    
    # End of Season
    dot.edge('UpdateTrust', 'End', label='End Season')
    dot.node('End', 'End Season')
    
    dot.render('trust_dynamics_over_season_flowchart', view=True)

def create_trust_recovery_post_crash_flowchart():
    dot = Digraph(comment='Trust Recovery Post Forecast Accuracy Crash')
    
    # Define start
    dot.node('Start', 'Start Simulation')
    
    # Step 1: Forecast Accuracy Crash
    dot.node('Crash', 'Forecast Accuracy Crash')
    dot.edge('Start', 'Crash')
    
    # Step 2: Trust Decreases
    dot.node('TrustDecreases', 'Trust Level Decreases')
    dot.edge('Crash', 'TrustDecreases')
    
    # Step 3: Government Intervention
    dot.node('Intervention', 'Government Intervention')
    dot.edge('TrustDecreases', 'Intervention', label='Policy Implementation')
    
    # Step 4: Trust Recovery
    dot.node('Recover', 'Trust Level Recovers')
    dot.edge('Intervention', 'Recover')
    
    # Step 5: Forecast Accuracy Restored
    dot.node('Restore', 'Forecast Accuracy Restored')
    dot.edge('Recover', 'Restore', label='Gradual Restoration')
    
    # End
    dot.edge('Restore', 'End', label='Recovery Complete')
    dot.node('End', 'End Simulation')
    
    dot.render('trust_recovery_post_crash_flowchart', view=True)

def create_yield_tracking_by_accuracy_flowchart():
    dot = Digraph(comment='Yield Tracking by Accuracy Level')
    
    # Define start
    dot.node('Start', 'Start Simulation')
    
    # Loop through seasons
    dot.node('Loop', 'For Each Season')
    dot.edge('Start', 'Loop')
    
    # Step 1: Provide Forecast
    dot.node('Forecast', 'Provide Forecast with Accuracy Level')
    dot.edge('Loop', 'Forecast')
    
    # Step 2: Farmer Decides Crop
    dot.node('DecideCrop', 'Farmer Decides Crop')
    dot.edge('Forecast', 'DecideCrop', label='Based on Forecast & Rules')
    
    # Add decision rules for crop selection
    dot.node('CropRule', 'Determine Crop Type\n(Rainfall & Risk Tolerance)')
    dot.edge('Forecast', 'CropRule', label='Evaluate Conditions')
    dot.edge('CropRule', 'DecideCrop', label='Select Crop')
    
    # Step 3: Calculate Yield with Rules
    dot.node('CalculateYield', 'Calculate Yield')
    dot.edge('DecideCrop', 'CalculateYield', label='Apply Yield Rules')
    
    # Step 4: Update Wealth and Yield
    dot.node('UpdateWealth', 'Update Wealth and Total Yield')
    dot.edge('CalculateYield', 'UpdateWealth')
    
    # Step 5: Track Yield
    dot.node('TrackYield', 'Track Average Yield')
    dot.edge('UpdateWealth', 'TrackYield')
    
    # Repeat Loop
    dot.edge('TrackYield', 'Loop', label='Next Season')
    
    # End
    dot.edge('Loop', 'End', style='dotted', label='Simulation End')
    dot.node('End', 'End Simulation')
    
    dot.render('yield_tracking_by_accuracy_flowchart', view=True)

if __name__ == "__main__":
    create_flowchart()
    create_trust_by_dissemination_mode_flowchart()
    create_trust_dynamics_over_season_flowchart()
    create_trust_recovery_post_crash_flowchart()
    create_yield_tracking_by_accuracy_flowchart()