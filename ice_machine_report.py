import pandas as pd
from datetime import datetime
import plotly.graph_objs as go
import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table
import numpy as np

# Initialize Dash app with external stylesheet
app = dash.Dash(__name__, assets_folder='assets', external_stylesheets=['assets/style.css?v=1.0'], suppress_callback_exceptions=True)

# Load data
assets = pd.read_csv('assets.csv')
service_work_orders = pd.read_csv('work_orders.csv')
installation_work_orders = pd.read_csv('installation_work_orders.csv')

# Convert date columns to datetime first
assets['manufacturing_date'] = pd.to_datetime(assets['manufacturing_date'], format='%m/%d/%Y', errors='coerce')
assets['decommissioned_date'] = pd.to_datetime(assets['decommissioned_date'], format='%m/%d/%Y', errors='coerce')
service_work_orders['schedule_date'] = pd.to_datetime(service_work_orders['schedule_date'], format='%m/%d/%Y', errors='coerce')
installation_work_orders['schedule_date'] = pd.to_datetime(installation_work_orders['schedule_date'], format='%m/%d/%Y', errors='coerce')

# Define time_span_years and current_date
current_date = pd.Timestamp(datetime(2025, 4, 1))  # Data pulled on 4/1/2025
start_date = pd.Timestamp(datetime(2023, 4, 1))   # First entry on 4/1/2023
num_years = 2  # We’re defining 2 full years: 4/1/2023-3/31/2024 and 4/1/2024-3/31/2025

# Define year periods
year_periods = {
    1: (pd.Timestamp(datetime(2023, 4, 1)), pd.Timestamp(datetime(2024, 3, 31))),
    2: (pd.Timestamp(datetime(2024, 4, 1)), pd.Timestamp(datetime(2025, 3, 31)))
}

# Function to assign a work order to a year period
def assign_year_period(schedule_date):
    if pd.isna(schedule_date):
        return None
    for year_num, (start, end) in year_periods.items():
        if start <= schedule_date <= end:
            return year_num
    return None

# Assign year periods to installation work orders
installation_work_orders['year_period'] = installation_work_orders['schedule_date'].apply(assign_year_period)
print("Sample installation work orders with year periods:", installation_work_orders[['schedule_date', 'year_period']].head().to_string())

# Ensure asset_number is string type before normalization
assets['asset_number'] = assets['asset_number'].astype(str)
service_work_orders['asset_number'] = service_work_orders['asset_number'].astype(str)
installation_work_orders['asset_number'] = installation_work_orders['asset_number'].astype(str)

# Normalize asset_number in all DataFrames
assets['asset_number'] = assets['asset_number'].str.strip().str.upper().str.replace('-', '')
service_work_orders['asset_number'] = service_work_orders['asset_number'].str.strip().str.upper().str.replace('-', '')
installation_work_orders['asset_number'] = installation_work_orders['asset_number'].str.strip().str.upper().str.replace('-', '')

# Standardize manufacturer names (remove quotes)
assets['manufacturer'] = assets['manufacturer'].str.strip('"')

# Debugging: Print sample asset_number values after normalization
print("Sample asset_number from assets.csv:", assets['asset_number'].head().tolist())
print("Sample asset_number from work_orders.csv:", service_work_orders['asset_number'].head().tolist())
print("Sample asset_number from installation_work_orders.csv:", installation_work_orders['asset_number'].head().tolist())

# Debugging: Check Manitowoc Ice asset_numbers in assets.csv
manitowoc_assets = assets[assets['manufacturer'] == 'Manitowoc Ice']
print("Total Manitowoc Ice assets in assets.csv:", len(manitowoc_assets))
print("Sample Manitowoc Ice asset_numbers in assets.csv:", manitowoc_assets['asset_number'].head().tolist())

# Create a mapping from asset_number to manufacturer
asset_to_manufacturer = assets.set_index('asset_number')['manufacturer'].to_dict()

# Add manufacturer to service_work_orders for debugging
service_work_orders['manufacturer'] = service_work_orders['asset_number'].map(asset_to_manufacturer)
# Debug: Check if service_work_orders actually contains Manitowoc entries
manitowoc_service_pre = service_work_orders[service_work_orders['manufacturer'] == 'Manitowoc Ice']
print("Manitowoc service work orders in raw data:", len(manitowoc_service_pre))
print("Sample Manitowoc service work orders in raw data:", manitowoc_service_pre[['asset_number', 'schedule_date']].head().to_string())

# Add manufacturer to installation_work_orders
installation_work_orders['manufacturer'] = installation_work_orders['asset_number'].map(asset_to_manufacturer)

# Debugging: Check Manitowoc Ice installation work orders
manitowoc_install = installation_work_orders[installation_work_orders['manufacturer'] == 'Manitowoc Ice']
print("Total Manitowoc Ice installation work orders:", len(manitowoc_install))
print("Sample Manitowoc Ice asset_numbers in installation_work_orders.csv:", manitowoc_install['asset_number'].head().tolist())

# Convert date columns to datetime
assets['manufacturing_date'] = pd.to_datetime(assets['manufacturing_date'], format='%m/%d/%Y', errors='coerce')
assets['decommissioned_date'] = pd.to_datetime(assets['decommissioned_date'], format='%m/%d/%Y', errors='coerce')
service_work_orders['schedule_date'] = pd.to_datetime(service_work_orders['schedule_date'], format='%m/%d/%Y', errors='coerce')
installation_work_orders['schedule_date'] = pd.to_datetime(installation_work_orders['schedule_date'], format='%m/%d/%Y', errors='coerce')

# Calculate active years with last service date consideration
assets['end_date'] = assets['decommissioned_date'].where(assets['decommissioned_date'].notna(), current_date)
# Align with user's method: 
# - Non-decommissioned assets before 4/1/2023 get 2 years
# - Non-decommissioned assets after 4/1/2023 get (4/1/2025 - manufacturing_date)
# - Decommissioned assets get (decommissioned_date - 4/1/2023)
def calculate_active_years(row):
    if pd.notna(row['decommissioned_date']):
        # Decommissioned: from 4/1/2023 to decommissioned_date
        start = start_date
        end = min(row['decommissioned_date'], current_date)
        return (end - start).days / 365.25
    else:
        # Non-decommissioned
        if pd.notna(row['manufacturing_date']) and row['manufacturing_date'] < start_date:
            # Before 4/1/2023: 2 years
            return 2.0
        else:
            # After 4/1/2023: from manufacturing_date to 4/1/2025
            start = max(row['manufacturing_date'], start_date) if pd.notna(row['manufacturing_date']) else start_date
            return (current_date - start).days / 365.25

assets['active_years'] = assets.apply(calculate_active_years, axis=1)
assets['active_years'] = assets['active_years'].clip(lower=0).replace(0, 0.001)  # Avoid division by zero

# Filter work orders to only include those within active periods
def filter_work_orders(work_orders, assets, work_order_type='service'):
    merged = work_orders.merge(assets[['asset_number', 'manufacturing_date', 'end_date']], on='asset_number', how='left')
    merged['manufacturing_date'] = merged['manufacturing_date'].fillna(pd.Timestamp('1900-01-01'))
    merged['end_date'] = merged['end_date'].fillna(current_date)
    manitowoc_pre = merged[merged['asset_number'].isin(manitowoc_assets['asset_number'])]
    print(f"Manitowoc {work_order_type} work orders before filter: {len(manitowoc_pre)}")
    filtered = merged[
        (merged['schedule_date'].isna()) | 
        ((merged['schedule_date'] >= merged['manufacturing_date']) & 
         (merged['schedule_date'] <= merged['end_date']))
    ]
    invalid = merged[merged['schedule_date'] > merged['end_date']]
    if not invalid.empty:
        print(f"Warning: {len(invalid)} {work_order_type} work orders found after end_date:")
        print(invalid[['asset_number', 'schedule_date', 'end_date']].head())
    manitowoc_post = filtered[filtered['asset_number'].isin(manitowoc_assets['asset_number'])]
    print(f"Manitowoc {work_order_type} work orders after filter: {len(manitowoc_post)}")
    print(f"Filtered {len(filtered)} {work_order_type} work orders from {len(work_orders)} total")
    return filtered

service_work_orders_active = filter_work_orders(service_work_orders, assets, 'service')
installation_work_orders_active = filter_work_orders(installation_work_orders, assets, 'installation')

# Adjust active_years based on last service date for decommissioned assets
last_service_date = service_work_orders_active.groupby('asset_number')['schedule_date'].max().reset_index()
assets = pd.merge(assets, last_service_date, on='asset_number', how='left')
assets = assets.rename(columns={'schedule_date': 'last_service_date'})
print("Columns in assets after merge:", assets.columns.tolist())
assets['end_date'] = assets.apply(
    lambda row: row['decommissioned_date'] if pd.notna(row['decommissioned_date']) 
    else (row['last_service_date'] if pd.notna(row['last_service_date']) 
          else current_date), axis=1
)

# Calculate ages
assets['age'] = (current_date - assets['manufacturing_date']).dt.days / 365.25
assets['age'] = assets['age'].fillna(0)
# Adjust bins to include exactly 5 years in the 5-10 years category
assets['age_category'] = pd.cut(assets['age'], bins=[0, 5.0001, 10, float('inf')], labels=['<5 Years', '5-10 Years', '10+ Years'])

# Debug IM-500SAA service work orders
im500saa_assets = assets[assets['asset_name'] == 'IM-500SAA']
print("Total IM-500SAA assets:", len(im500saa_assets))
im500saa_service = service_work_orders_active[service_work_orders_active['asset_number'].isin(im500saa_assets['asset_number'])]
print("Total IM-500SAA service work orders:", len(im500saa_service))
# Debug manufacturing dates distribution
im500saa_before_2023 = im500saa_assets[im500saa_assets['manufacturing_date'] < start_date]
im500saa_after_2023 = im500saa_assets[im500saa_assets['manufacturing_date'] >= start_date]
print("IM-500SAA assets manufactured before 4/1/2023:", len(im500saa_before_2023))
print("IM-500SAA assets manufactured on or after 4/1/2023:", len(im500saa_after_2023))
if not im500saa_after_2023.empty:
    print("Sample IM-500SAA assets manufactured on or after 4/1/2023:", im500saa_after_2023[['asset_number', 'manufacturing_date', 'active_years']].head().to_string())
total_active_years = im500saa_assets['active_years'].sum()
print("Total active years for IM-500SAA assets:", total_active_years)
if total_active_years > 0:
    avg_service_per_year = len(im500saa_service) / total_active_years
    print("Average service work orders per year for IM-500SAA (total WO / total active years):", avg_service_per_year)
    # Debug decommissioned assets
    im500saa_decommissioned = im500saa_assets[im500saa_assets['decommissioned_date'].notna()]
    print("Number of decommissioned IM-500SAA assets:", len(im500saa_decommissioned))
    total_decommissioned_years = im500saa_decommissioned['active_years'].sum()
    print("Total active years for decommissioned IM-500SAA assets:", total_decommissioned_years)
    if not im500saa_decommissioned.empty:
        print("Sample decommissioned IM-500SAA assets:", im500saa_decommissioned[['asset_number', 'manufacturing_date', 'decommissioned_date', 'active_years']].head().to_string())
    # Debug non-decommissioned assets
    im500saa_non_decommissioned = im500saa_assets[im500saa_assets['decommissioned_date'].isna()]
    print("Number of non-decommissioned IM-500SAA assets:", len(im500saa_non_decommissioned))
    total_non_decommissioned_years = im500saa_non_decommissioned['active_years'].sum()
    print("Total active years for non-decommissioned IM-500SAA assets:", total_non_decommissioned_years)
    if not im500saa_non_decommissioned.empty:
        print("Sample non-decommissioned IM-500SAA assets:", im500saa_non_decommissioned[['asset_number', 'manufacturing_date', 'active_years']].head().to_string())
else:
    print("No active years for IM-500SAA assets")

# Debug IM-500SAA installation work orders for <5 years
im500saa_under_5 = im500saa_assets[im500saa_assets['age_category'] == '<5 Years']
print("Total IM-500SAA assets under 5 years:", len(im500saa_under_5))
im500saa_install_under_5 = installation_work_orders_active[installation_work_orders_active['asset_number'].isin(im500saa_under_5['asset_number'])]
print("Total IM-500SAA installation work orders under 5 years:", len(im500saa_install_under_5))
if not im500saa_install_under_5.empty:
    im500saa_install_under_5 = im500saa_install_under_5.copy()  # Create a copy to avoid SettingWithCopyWarning
    im500saa_install_under_5.loc[:, 'year'] = im500saa_install_under_5['schedule_date'].dt.year
    years = im500saa_install_under_5['year'].dropna().unique()
    print("Years of IM-500SAA installation work orders under 5 years:", years)
    time_span = 2  # Hardcode to 2 years (2023 to 2025)
    avg_installations = len(im500saa_install_under_5) / time_span
    print("Average installations per year for IM-500SAA under 5 years:", avg_installations)
else:
    print("No installation work orders for IM-500SAA under 5 years")

# Calculate work order counts using size() to avoid NaT issues
service_counts = service_work_orders_active.groupby('asset_number').size().reset_index(name='service_number_of_work_orders')
# Debug service counts for Manitowoc
manitowoc_service_counts = service_counts[service_counts['asset_number'].isin(manitowoc_assets['asset_number'])]
print("Manitowoc service counts before merge:", len(manitowoc_service_counts))
print("Sample Manitowoc service counts:", manitowoc_service_counts.head().to_string())
installation_counts = installation_work_orders_active.groupby('asset_number').size().reset_index(name='installation_number_of_work_orders')

# Merge counts into assets DataFrame
assets = pd.merge(assets, service_counts, on='asset_number', how='left')
assets = pd.merge(assets, installation_counts, on='asset_number', how='left')

# Fill NaN values with 0
assets['service_number_of_work_orders'] = assets['service_number_of_work_orders'].fillna(0)
assets['installation_number_of_work_orders'] = assets['installation_number_of_work_orders'].fillna(0)

# Calculate averages for Service and totals for Installation
assets['service_average_work_orders_per_year'] = assets['service_number_of_work_orders'] / assets['active_years']

# Debugging: Check Manitowoc Ice assets
manitowoc_assets = assets[assets['manufacturer'] == 'Manitowoc Ice']
manitowoc_with_service = manitowoc_assets[manitowoc_assets['service_number_of_work_orders'] > 0]
print("Number of Manitowoc Ice assets with service work orders:", len(manitowoc_with_service))
print("Sample Manitowoc assets with service:", manitowoc_with_service[['asset_number', 'service_number_of_work_orders']].head().to_string())

# Ensure 'size' is numeric
assets['size'] = pd.to_numeric(assets['size'], errors='coerce')

# Calculate size range for Ice Maker assets
ice_makers = assets[assets['category'] == 'Ice Maker']
ice_maker_size_min = ice_makers['size'].min() if not ice_makers['size'].isna().all() else 0
ice_maker_size_max = ice_makers['size'].max() if not ice_makers['size'].isna().all() else 1000

# Define placeholder figures
empty_fig = go.Figure()
empty_fig.update_layout(
    title='No Data Available',
    title_x=0.5,
    plot_bgcolor='#222222',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='#E0E0E0',
    height=400
)

placeholder_fig = go.Figure()
placeholder_fig.update_layout(
    title='Please select a model to view the data',
    title_x=0.5,
    plot_bgcolor='#222222',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='#E0E0E0',
    height=500
)

# App layout
app.layout = html.Div([
    html.Meta(httpEquiv="Cache-Control", content="no-cache, no-store, must-revalidate"),
    html.Meta(httpEquiv="Pragma", content="no-cache"),
    html.Meta(httpEquiv="Expires", content="0"),
    
    html.Div([
        dcc.Tabs(
            id='data-toggle',
            value='service',
            children=[
                dcc.Tab(label='Service', value='service'),
                dcc.Tab(label='Installation', value='installation'),
            ],
            className='toggle-tabs',
            style={'display': 'inline-flex', 'backgroundColor': '#2A2A2A', 'borderRadius': '25px', 'padding': '3px'}
        ),
        html.Div(
            html.H1("Empire Asset Report", style={'color': '#E0E0E0', 'fontFamily': 'Arial'}),
            className='header-title'
        ),
        html.Img(src='/assets/company_logo.png', className='logo')
    ], className='header'),
    dcc.Tabs(id='tabs', value='overview', children=[
        dcc.Tab(label='Overview', value='overview'),
        dcc.Tab(label='By Model', value='by-model'),
        dcc.Tab(label='By Manufacturer', value='comparison'),
    ]),
    html.Div(id='tabs-content')
], style={'backgroundColor': '#1E1E1E', 'height': '100vh'})

# Callback to render tab content
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    print(f"Rendering tab: {tab} - Starting render_content")
    if tab == 'overview':
        print("Rendering Overview tab")
        return html.Div([
            html.H2("Overview", style={'color': '#E0E0E0'}),
            html.Div([
                html.Label("Equipment Category:", style={'fontWeight': 'bold', 'color': '#E0E0E0', 'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='category-dropdown',
                    options=[{'label': 'All Categories', 'value': 'all'}] + 
                            [{'label': category, 'value': category} for category in assets['category'].unique()],
                    value='all',
                    placeholder="Select a category",
                    style={'backgroundColor': '#404040', 'color': '#666666', 'width': '200px', 'display': 'inline-block', 'verticalAlign': 'middle'}
                ),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
            html.Div([
                html.Label("Size Range Min:", style={'fontWeight': 'bold', 'color': '#E0E0E0'}),
                dcc.Input(id='overview-min-size', type='number', value=ice_maker_size_min, min=ice_maker_size_min, max=ice_maker_size_max, step=1, debounce=True,
                          style={'backgroundColor': '#333333', 'color': '#E0E0E0', 'width': '100px'}),
                html.Label("Max:", style={'fontWeight': 'bold', 'color': '#E0E0E0', 'marginLeft': '10px'}),
                dcc.Input(id='overview-max-size', type='number', value=ice_maker_size_max, min=ice_maker_size_min, max=ice_maker_size_max, step=1, debounce=True,
                          style={'backgroundColor': '#333333', 'color': '#E0E0E0', 'width': '100px', 'marginLeft': '10px'}),
            ], style={'marginBottom': '10px'}),
            html.Div(id='overview-size-message', style={'color': '#A0A0A0', 'marginBottom': '10px'}),
            dcc.Graph(id='overview-main-graph'),
            dash_table.DataTable(id='overview-main-table', style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'backgroundColor': '#222222', 'color': '#E0E0E0'}),
            html.Div([
                html.Div([dcc.Graph(id='overview-less-5-graph'), dash_table.DataTable(id='overview-less-5-table', style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'backgroundColor': '#222222', 'color': '#E0E0E0'})], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='overview-5-10-graph'), dash_table.DataTable(id='overview-5-10-table', style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'backgroundColor': '#222222', 'color': '#E0E0E0'})], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='overview-10-plus-graph'), dash_table.DataTable(id='overview-10-plus-table', style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'backgroundColor': '#222222', 'color': '#E0E0E0'})], style={'width': '33%', 'display': 'inline-block'})
            ])
        ])
    elif tab == 'by-model':
        print("Rendering By Model tab")
        return html.Div([
            html.H2("By Model", style={'color': '#E0E0E0'}),
            dcc.Dropdown(
                id='model-dropdown',
                options=[{'label': model, 'value': model} for model in assets['asset_name'].unique()],
                value=None,
                placeholder="Select a model",
                style={'backgroundColor': '#404040', 'color': '#666666'}
            ),
            html.Div(id='model-avg-text', style={'color': '#E0E0E0', 'fontSize': '20px', 'marginTop': '20px'}),
            html.Div(id='asset-count-text', style={'color': '#E0E0E0', 'fontSize': '20px', 'marginTop': '5px'}),
            dcc.Graph(id='model-grouped-graph', figure=placeholder_fig, style={'width': '80%', 'margin': '0 auto'}),
            dash_table.DataTable(id='model-data-table', style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'backgroundColor': '#222222', 'color': '#E0E0E0'}),
            dcc.Graph(id='model-stddev-graph', figure=placeholder_fig, style={'width': '80%', 'margin': '0 auto'}),
            dash_table.DataTable(id='model-stddev-table', style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'backgroundColor': '#222222', 'color': '#E0E0E0'})
        ])
    elif tab == 'comparison':
        print("Rendering By Manufacturer tab")
        return html.Div([
            html.H2("By Manufacturer", style={'color': '#E0E0E0'}),
            html.Div([
                html.Label("Choose a Manufacturer:", style={'fontWeight': 'bold', 'color': '#E0E0E0'}),
                dcc.Dropdown(
                    id='manufacturer-dropdown',
                    options=[{'label': m, 'value': m} for m in assets['manufacturer'].unique()],
                    value=None,
                    placeholder="Select a manufacturer",
                    style={'backgroundColor': '#404040', 'color': '#666666'}
                ),
            ], style={'marginBottom': '20px'}),
            html.P(f"Valid size range: {ice_maker_size_min} to {ice_maker_size_max}", style={'color': '#E0E0E0'}),
            html.Div([
                html.Label("Size Range Min:", style={'fontWeight': 'bold', 'color': '#E0E0E0'}),
                dcc.Input(id='min-size', type='number', value=ice_maker_size_min, min=ice_maker_size_min, max=ice_maker_size_max, step=1, debounce=True,
                          style={'backgroundColor': '#333333', 'color': '#E0E0E0', 'width': '100px'}),
                html.Label("Max:", style={'fontWeight': 'bold', 'color': '#E0E0E0', 'marginLeft': '10px'}),
                dcc.Input(id='max-size', type='number', value=ice_maker_size_max, min=ice_maker_size_min, max=ice_maker_size_max, step=1, debounce=True,
                          style={'backgroundColor': '#333333', 'color': '#E0E0E0', 'width': '100px', 'marginLeft': '10px'})
            ], style={'marginBottom': '20px'}),
            html.Div(id='size-range-message', style={'color': '#A0A0A0', 'marginBottom': '10px'}),
            dcc.Graph(id='comparison-total-graph'),
            dash_table.DataTable(id='comparison-main-table', style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'backgroundColor': '#222222', 'color': '#E0E0E0'}),
            html.Div([
                html.Div([dcc.Graph(id='comparison-less-5-graph'), dash_table.DataTable(id='comparison-less-5-table', style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'backgroundColor': '#222222', 'color': '#E0E0E0'})], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='comparison-5-10-graph'), dash_table.DataTable(id='comparison-5-10-table', style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'backgroundColor': '#222222', 'color': '#E0E0E0'})], style={'width': '33%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='comparison-10-plus-graph'), dash_table.DataTable(id='comparison-10-plus-table', style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'backgroundColor': '#222222', 'color': '#E0E0E0'})], style={'width': '33%', 'display': 'inline-block'})
            ])
        ])
    else:
        print(f"Unknown tab: {tab}")
        return html.Div("Unknown tab selected", style={'color': '#E0E0E0'})
    
# Callback for Overview tab
@app.callback(
    [Output('overview-main-graph', 'figure'),
     Output('overview-main-table', 'data'),
     Output('overview-main-table', 'columns'),
     Output('overview-less-5-graph', 'figure'),
     Output('overview-less-5-table', 'data'),
     Output('overview-less-5-table', 'columns'),
     Output('overview-5-10-graph', 'figure'),
     Output('overview-5-10-table', 'data'),
     Output('overview-5-10-table', 'columns'),
     Output('overview-10-plus-graph', 'figure'),
     Output('overview-10-plus-table', 'data'),
     Output('overview-10-plus-table', 'columns'),
     Output('overview-size-message', 'children')],
    [Input('tabs', 'value'),
     Input('data-toggle', 'value'),
     Input('category-dropdown', 'value'),
     Input('overview-min-size', 'value'),
     Input('overview-max-size', 'value')]
)
def update_overview_graphs(tab, toggle, selected_category, min_size, max_size):
    if tab != 'overview':
        raise dash.exceptions.PreventUpdate
    
    if selected_category == 'all':
        filtered_data = assets
    else:
        filtered_data = assets[assets['category'] == selected_category]

    min_size = float(min_size if min_size is not None else ice_maker_size_min)
    max_size = float(max_size if max_size is not None else ice_maker_size_max)
    if max_size < min_size:
        max_size = min_size
    filtered_data = filtered_data[(filtered_data['size'].notna()) & 
                                 (filtered_data['size'] >= min_size) & 
                                 (filtered_data['size'] <= max_size)]
    
    message = f"Showing {len(filtered_data)} assets in size range {min_size}–{max_size}."

    if toggle == 'service':
        avg_column = 'service_average_work_orders_per_year'
        title = 'Average Service Work Orders per Year by Manufacturer'
        yaxis_title = 'Work Orders per Year'
        overview_data = filtered_data.groupby('manufacturer')[avg_column].mean().reset_index()
        overview_data[avg_column] = overview_data[avg_column].round(2)
    else:
        avg_column = 'installs per year'
        title = 'Average Installs per Year by Manufacturer'
        yaxis_title = 'Installs per Year'
        # Merge with installation work orders to count per year
        merged = filtered_data.merge(installation_work_orders[['asset_number', 'year_period']], on='asset_number', how='left')
        # Group by manufacturer and year_period to get total installations
        yearly_counts = merged.groupby(['manufacturer', 'year_period']).size().reset_index(name='install_count')
        # Calculate average installs per year
        overview_data = yearly_counts.groupby('manufacturer')['install_count'].sum().reset_index()
        overview_data[avg_column] = (overview_data['install_count'] / num_years).round(2)
        overview_data = overview_data.drop(columns=['install_count'])  # Remove install_count column

    asset_counts = filtered_data.groupby('manufacturer').size().reset_index(name='asset_count')
    overview_data = overview_data.merge(asset_counts, on='manufacturer')
    overview_data = overview_data.sort_values('asset_count', ascending=False)
    hover_text = [f"Assets: {count}" for count in overview_data['asset_count']]
    fig_main = go.Figure(data=[go.Bar(
        x=overview_data['manufacturer'],
        y=overview_data[avg_column],
        hovertext=hover_text,
        hoverinfo='text+y'
    )])
    fig_main.update_layout(
        title=title,
        title_x=0.5,
        plot_bgcolor='#222222',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#E0E0E0',
        height=400,
        yaxis_title=yaxis_title
    )
    top_manufacturers = asset_counts.nlargest(5, 'asset_count')['manufacturer'].tolist()
    age_groups = ['<5 Years', '5-10 Years', '10+ Years']
    figs_age = []
    tables_age = []
    for age in age_groups:
        age_data = filtered_data[filtered_data['age_category'] == age]
        if toggle == 'service':
            age_avg = age_data.groupby('manufacturer')[avg_column].mean().reset_index()
            age_avg[avg_column] = age_avg[avg_column].round(2)
        else:
            # Merge with installation work orders to count per year
            merged_age = age_data.merge(installation_work_orders[['asset_number', 'year_period']], on='asset_number', how='left')
            yearly_counts_age = merged_age.groupby(['manufacturer', 'year_period']).size().reset_index(name='install_count')
            age_avg = yearly_counts_age.groupby('manufacturer')['install_count'].sum().reset_index()
            age_avg[avg_column] = (age_avg['install_count'] / num_years).round(2)
            age_avg = age_avg.drop(columns=['install_count'])  # Remove install_count column
        age_counts = age_data.groupby('manufacturer').size().reset_index(name='asset_count')
        age_avg = age_avg.merge(age_counts, on='manufacturer', how='left')
        age_avg['asset_count'] = age_avg['asset_count'].fillna(0)
        age_avg = age_avg[age_avg['manufacturer'].isin(top_manufacturers)]
        age_avg = age_avg.set_index('manufacturer').reindex(top_manufacturers).fillna(0).reset_index()
        hover_text_age = [f"Assets: {count}" for count in age_avg['asset_count']]
        fig = go.Figure(data=[go.Bar(
            x=age_avg['manufacturer'],
            y=age_avg[avg_column],
            hovertext=hover_text_age,
            hoverinfo='text+y'
        )])
        fig.update_layout(
            title=age,
            title_x=0.5,
            plot_bgcolor='#222222',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#E0E0E0',
            height=300,
            yaxis_title=yaxis_title
        )
        figs_age.append(fig)
        tables_age.append(age_avg)
    
    return [
        fig_main,
        overview_data.to_dict('records'),
        [{"name": i, "id": i} for i in overview_data.columns],
        figs_age[0],
        tables_age[0].to_dict('records'),
        [{"name": i, "id": i} for i in tables_age[0].columns],
        figs_age[1],
        tables_age[1].to_dict('records'),
        [{"name": i, "id": i} for i in tables_age[1].columns],
        figs_age[2],
        tables_age[2].to_dict('records'),
        [{"name": i, "id": i} for i in tables_age[2].columns],
        message
    ]

# Callback for By Model tab
@app.callback(
    [Output('model-avg-text', 'children'),
     Output('asset-count-text', 'children'),
     Output('model-grouped-graph', 'figure'),
     Output('model-data-table', 'data'),
     Output('model-data-table', 'columns'),
     Output('model-stddev-graph', 'figure'),
     Output('model-stddev-table', 'data'),
     Output('model-stddev-table', 'columns')],
    [Input('model-dropdown', 'value'),
     Input('data-toggle', 'value')]
)
def update_model_content(selected_model, toggle):
    if selected_model is None:
        return "Select a model", "", placeholder_fig, [], [], placeholder_fig, [], []
    
    model_data = assets[assets['asset_name'] == selected_model]
    asset_count = len(model_data)
    asset_count_text = f"Asset Count: {asset_count}"
    
    if toggle == 'service':
        work_orders = service_work_orders_active
        work_order_type = "Service Work Orders"
        total_work_orders = len(work_orders[work_orders['asset_number'].isin(model_data['asset_number'])])
        total_active_years = model_data['active_years'].sum()
        if total_active_years > 0:
            avg_value = round(total_work_orders / total_active_years, 2)
        else:
            avg_value = 0
        avg_text = f"Average {work_order_type} per Year: {avg_value}"
    else:
        work_orders = installation_work_orders_active
        work_order_type = "Installations"
        total_work_orders = len(work_orders[work_orders['asset_number'].isin(model_data['asset_number'])])
        avg_value = round(total_work_orders / num_years, 2)
        avg_text = f"Average {work_order_type} per Year: {avg_value}"

    model_work_orders = work_orders[work_orders['asset_number'].isin(model_data['asset_number'])].copy()
    
    age_groups = ['<5 Years', '5-10 Years', '10+ Years']  # Fixed order
    color_map = {'<5 Years': '#1f77b4', '5-10 Years': '#2ca02c', '10+ Years': '#d62728'}
    fig = go.Figure()
    table_data = []
    stddev_fig = go.Figure()
    stddev_table_data = []
    
    for age in age_groups:  # Always include all age groups
        age_data = model_data[model_data['age_category'] == age]
        age_work_orders = model_work_orders[model_work_orders['asset_number'].isin(age_data['asset_number'])]
        
        # Debug IM-500SAB specifically
        if selected_model == 'IM-500SAB':
            print(f"IM-500SAB {age}: {len(age_data)} assets")
            print(f"IM-500SAB {age} {work_order_type.lower()} work orders: {len(age_work_orders)}")
            if not age_work_orders.empty:
                if toggle == 'service':
                    print(f"Years of IM-500SAB {age} service work orders:", age_work_orders['schedule_date'].dt.year.unique())
                else:
                    print(f"Year periods of IM-500SAB {age} installation work orders:", age_work_orders['year_period'].unique())
        
        asset_count_for_group = len(age_data)  # Calculate asset count for the age group
        
        if toggle == 'service':
            # Use pre-calculated service_average_work_orders_per_year
            if not age_data.empty:
                # Filter out negative values
                avg_per_asset = age_data['service_average_work_orders_per_year']
                avg_per_asset = avg_per_asset[avg_per_asset >= 0]
                if not avg_per_asset.empty:
                    min_avg = avg_per_asset.min()
                    max_avg = avg_per_asset.max()
                    avg_avg = avg_per_asset.mean()
                    # Calculate standard deviation
                    stddev = avg_per_asset.std() if len(avg_per_asset) > 1 else 0
                    # Calculate 68% and 95% ranges
                    range_68_lower = max(0, avg_avg - stddev)  # Clip negative values to 0
                    range_68_upper = avg_avg + stddev
                    range_95_lower = max(0, avg_avg - 2 * stddev)  # Clip negative values to 0
                    range_95_upper = avg_avg + 2 * stddev
                    # Create histogram for standard deviation graph
                    stddev_fig.add_trace(go.Histogram(
                        x=avg_per_asset,
                        name=age,
                        marker_color=color_map[age],
                        opacity=0.75,
                        histnorm='probability density',
                        showlegend=True
                    ))
                    # Add normal distribution curve with increased line width
                    if stddev > 0:  # Only add curve if stddev is non-zero
                        x_range = np.linspace(min(avg_per_asset), max(avg_per_asset), 100)
                        y = (1 / (stddev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - avg_avg) / stddev) ** 2)
                        stddev_fig.add_trace(go.Scatter(
                            x=x_range,
                            y=y,
                            mode='lines',
                            name=f'{age} Normal',
                            line=dict(color=color_map[age], width=3),  # Increased line width
                            showlegend=True
                        ))
                else:
                    min_avg = max_avg = avg_avg = 0
                    stddev = 0
                    range_68_lower = range_68_upper = range_95_lower = range_95_upper = 0
            else:
                min_avg = max_avg = avg_avg = 0
                stddev = 0
                range_68_lower = range_68_upper = range_95_lower = range_95_upper = 0
        else:
            if not age_work_orders.empty:
                # Calculate total installations per year period
                yearly_counts = age_work_orders.groupby('year_period').size().reset_index(name='total_work_orders')
                # Ensure all year periods are present
                all_years = pd.DataFrame({'year_period': range(1, num_years + 1)})
                yearly_counts = all_years.merge(yearly_counts, on='year_period', how='left').fillna(0)
                # Debug the yearly counts
                if selected_model == 'IM-500SAB':
                    print(f"Yearly installation counts for IM-500SAB {age}: {yearly_counts[['year_period', 'total_work_orders']].to_dict('records')}")
                min_count = yearly_counts['total_work_orders'].min()
                max_count = yearly_counts['total_work_orders'].max()
                total_installations = yearly_counts['total_work_orders'].sum()
                avg_count = total_installations / num_years
                # Calculate standard deviation of yearly counts
                stddev = yearly_counts['total_work_orders'].std() if len(yearly_counts) > 1 else 0
                # Calculate 68% and 95% ranges
                range_68_lower = max(0, avg_count - stddev)  # Clip negative values to 0
                range_68_upper = avg_count + stddev
                range_95_lower = max(0, avg_count - 2 * stddev)  # Clip negative values to 0
                range_95_upper = avg_count + 2 * stddev
                # Create histogram for standard deviation graph
                stddev_fig.add_trace(go.Histogram(
                    x=yearly_counts['total_work_orders'],
                    name=age,
                    marker_color=color_map[age],
                    opacity=0.75,
                    histnorm='probability density',
                    showlegend=True
                ))
                # Add normal distribution curve with increased line width
                if stddev > 0:  # Only add curve if stddev is non-zero
                    x_range = np.linspace(min(yearly_counts['total_work_orders']), max(yearly_counts['total_work_orders']), 100)
                    y = (1 / (stddev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - avg_count) / stddev) ** 2)
                    stddev_fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y,
                        mode='lines',
                        name=f'{age} Normal',
                        line=dict(color=color_map[age], width=3),  # Increased line width
                        showlegend=True
                    ))
            else:
                min_count = max_count = avg_count = 0
                stddev = 0
                range_68_lower = range_68_upper = range_95_lower = range_95_upper = 0
        
        # Add traces to the main graph even if values are 0
        fig.add_trace(go.Scatter(
            x=[age],
            y=[min_count if toggle == 'installation' else min_avg],
            mode='markers',
            name=f'{age} Min',
            marker=dict(color=color_map[age], size=15, symbol='circle'),
            hovertemplate=f'{age} Min: {min_count if toggle == "installation" else min_avg:.2f}'
        ))
        fig.add_trace(go.Scatter(
            x=[age],
            y=[avg_count if toggle == 'installation' else avg_avg],
            mode='markers',
            name=f'{age} Avg',
            marker=dict(color=color_map[age], size=15, symbol='square'),
            hovertemplate=f'{age} Avg: {avg_count if toggle == "installation" else avg_avg:.2f}'
        ))
        fig.add_trace(go.Scatter(
            x=[age],
            y=[max_count if toggle == 'installation' else max_avg],
            mode='markers',
            name=f'{age} Max',
            marker=dict(color=color_map[age], size=15, symbol='circle'),
            hovertemplate=f'{age} Max: {max_count if toggle == "installation" else max_avg:.2f}'
        ))
        fig.add_trace(go.Scatter(
            x=[age, age, age],
            y=[min_count if toggle == 'installation' else min_avg, 
               avg_count if toggle == 'installation' else avg_avg, 
               max_count if toggle == 'installation' else max_avg],
            mode='lines',
            line=dict(color=color_map[age], width=4),
            showlegend=False
        ))
        
        # Add to main table data with asset count
        table_data.append({
            'Age Group': age,
            'Asset Count': asset_count_for_group,
            'Min': round(min_count if toggle == 'installation' else min_avg, 2),
            'Avg': round(avg_count if toggle == 'installation' else avg_avg, 2),
            'Max': round(max_count if toggle == 'installation' else max_avg, 2)
        })
        
        # Add to standard deviation table data with asset count
        stddev_table_data.append({
            'Age Group': age,
            'Asset Count': asset_count_for_group,
            'Standard Deviation': round(stddev, 2),
            '68% Range': f"{round(range_68_lower, 2)} - {round(range_68_upper, 2)}",
            '95% Range': f"{round(range_95_lower, 2)} - {round(range_95_upper, 2)}"
        })

    # Update layout for the main graph
    title = f"Min, Avg, Max {work_order_type} per Year for {selected_model}"
    yaxis_title = 'Work Orders per Year' if toggle == 'service' else 'Installs per Year'
    fig.update_layout(
        title=title,
        xaxis_title='Age Category',
        yaxis_title=yaxis_title,
        plot_bgcolor='#222222',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#E0E0E0',
        title_x=0.5,
        xaxis={'categoryorder': 'array', 'categoryarray': age_groups},
        height=500,
        annotations=[
            dict(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text="For Service: Min, Avg, Max of yearly average work orders per year. For Installations: Min, Avg, Max of yearly installation counts.",
                showarrow=False,
                font=dict(size=10, color='#E0E0E0'),
                align="center"
            )
        ]
    )
    
    # Update layout for the standard deviation graph
    stddev_title = f"Distribution of {work_order_type} per Year for {selected_model}"
    stddev_fig.update_layout(
        title=stddev_title,
        xaxis_title='Work Orders per Year' if toggle == 'service' else 'Installs per Year',
        yaxis_title='Density',
        plot_bgcolor='#222222',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#E0E0E0',
        title_x=0.5,
        barmode='overlay',
        height=500,
        showlegend=True
    )
    
    return avg_text, asset_count_text, fig, table_data, [{"name": i, "id": i} for i in ['Age Group', 'Asset Count', 'Min', 'Avg', 'Max']], stddev_fig, stddev_table_data, [{"name": i, "id": i} for i in ['Age Group', 'Asset Count', 'Standard Deviation', '68% Range', '95% Range']]

# Callback for By Manufacturer tab
@app.callback(
    [Output('comparison-total-graph', 'figure'),
     Output('comparison-main-table', 'data'),
     Output('comparison-main-table', 'columns'),
     Output('comparison-less-5-graph', 'figure'),
     Output('comparison-less-5-table', 'data'),
     Output('comparison-less-5-table', 'columns'),
     Output('comparison-5-10-graph', 'figure'),
     Output('comparison-5-10-table', 'data'),
     Output('comparison-5-10-table', 'columns'),
     Output('comparison-10-plus-graph', 'figure'),
     Output('comparison-10-plus-table', 'data'),
     Output('comparison-10-plus-table', 'columns'),
     Output('size-range-message', 'children')],
    [Input('tabs', 'value'),
     Input('manufacturer-dropdown', 'value'),
     Input('min-size', 'value'),
     Input('max-size', 'value'),
     Input('data-toggle', 'value')],
    [State('tabs', 'value')]
)
def update_by_manufacturer_graphs(tab, selected_manufacturer, min_size, max_size, toggle, current_tab):
    print(f"Updating By Manufacturer tab with manufacturer: {selected_manufacturer}, toggle: {toggle}, current_tab: {current_tab}")
    if tab != 'comparison' or current_tab != 'comparison':
        print("Preventing update: Not on comparison tab")
        raise dash.exceptions.PreventUpdate
    
    # Define the columns based on the toggle
    if toggle == 'service':
        avg_column = 'service_average_work_orders_per_year'
        table_columns = ['asset_name', 'service_average_work_orders_per_year', 'asset_count']
    else:
        avg_column = 'installs per year'
        table_columns = ['asset_name', 'installs per year', 'asset_count']
    
    if selected_manufacturer is None:
        print("No manufacturer selected, returning placeholder")
        empty_df = pd.DataFrame(columns=table_columns)
        empty_columns = [{"name": i, "id": i} for i in empty_df.columns]
        return [empty_fig, [], empty_columns, empty_fig, [], empty_columns, empty_fig, [], empty_columns, empty_fig, [], empty_columns, "Please select a manufacturer"]
    
    min_size = float(min_size if min_size is not None else ice_maker_size_min)
    max_size = float(max_size if max_size is not None else ice_maker_size_max)
    if max_size < min_size:
        max_size = min_size
    filtered_data = assets[(assets['manufacturer'] == selected_manufacturer) & 
                          (assets['category'] == 'Ice Maker') & 
                          (assets['size'].notna()) & 
                          (assets['size'] >= min_size) & 
                          (assets['size'] <= max_size)]
    if filtered_data.empty:
        message = f"No Ice Maker assets found for {selected_manufacturer} in size range {min_size}–{max_size}."
        print(f"No data for {selected_manufacturer}, returning empty figure")
        empty_df = pd.DataFrame(columns=table_columns)
        empty_columns = [{"name": i, "id": i} for i in empty_df.columns]
        return [empty_fig, [], empty_columns, empty_fig, [], empty_columns, empty_fig, [], empty_columns, empty_fig, [], empty_columns, message]
    else:
        message = f"Showing {len(filtered_data)} Ice Maker assets for {selected_manufacturer} in size range {min_size}–{max_size}."
    
    if toggle == 'service':
        avg_column = 'service_average_work_orders_per_year'
        title = f'Average Service Work Orders per Year by Asset Name for {selected_manufacturer}'
        yaxis_title = 'Work Orders per Year'
        total_avg = filtered_data.groupby('asset_name')[avg_column].mean().reset_index()
        total_avg[avg_column] = total_avg[avg_column].round(2)
    else:
        avg_column = 'installs per year'
        title = f'Average Installs per Year by Asset Name for {selected_manufacturer}'
        yaxis_title = 'Installs per Year'
        # Merge with installation work orders to count per year
        merged = filtered_data.merge(installation_work_orders[['asset_number', 'year_period']], on='asset_number', how='left')
        yearly_counts = merged.groupby(['asset_name', 'year_period']).size().reset_index(name='install_count')
        total_avg = yearly_counts.groupby('asset_name')['install_count'].sum().reset_index()
        total_avg[avg_column] = (total_avg['install_count'] / num_years).round(2)
        total_avg = total_avg.drop(columns=['install_count'])  # Remove install_count column

    asset_counts = filtered_data.groupby('asset_name').size().reset_index(name='asset_count')
    total_avg = total_avg.merge(asset_counts, on='asset_name')
    total_avg = total_avg.sort_values('asset_count', ascending=False)
    hover_text = [f"Assets: {count}" for count in total_avg['asset_count']]
    fig_total = go.Figure(data=[go.Bar(
        x=total_avg['asset_name'],
        y=total_avg[avg_column],
        hovertext=hover_text,
        hoverinfo='text+y'
    )])
    fig_total.update_layout(
        title=title,
        title_x=0.5,
        plot_bgcolor='#222222',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#E0E0E0',
        height=400,
        yaxis_title=yaxis_title
    )
    top_asset_names = asset_counts.nlargest(5, 'asset_count')['asset_name'].tolist()
    age_groups = ['<5 Years', '5-10 Years', '10+ Years']
    figs_age = []
    tables_age = []
    for age in age_groups:
        age_data = filtered_data[filtered_data['age_category'] == age]
        if age_data.empty:
            # If no data for this age group, create an empty DataFrame with the correct columns
            age_avg = pd.DataFrame(columns=table_columns)
        else:
            if toggle == 'service':
                age_avg = age_data.groupby('asset_name')[avg_column].mean().reset_index()
                age_avg[avg_column] = age_avg[avg_column].round(2)
            else:
                # Merge with installation work orders to count per year
                merged_age = age_data.merge(installation_work_orders[['asset_number', 'year_period']], on='asset_number', how='left')
                yearly_counts_age = merged_age.groupby(['asset_name', 'year_period']).size().reset_index(name='install_count')
                age_avg = yearly_counts_age.groupby('asset_name')['install_count'].sum().reset_index()
                age_avg[avg_column] = (age_avg['install_count'] / num_years).round(2)
                age_avg = age_avg.drop(columns=['install_count'])  # Remove install_count column
            age_counts = age_data.groupby('asset_name').size().reset_index(name='asset_count')
            age_avg = age_avg.merge(age_counts, on='asset_name', how='left')
            age_avg['asset_count'] = age_avg['asset_count'].fillna(0)
            age_avg = age_avg[age_avg['asset_name'].isin(top_asset_names)]
            age_avg = age_avg.set_index('asset_name').reindex(top_asset_names).fillna(0).reset_index()
        hover_text_age = [f"Assets: {count}" for count in age_avg['asset_count']]
        fig = go.Figure(data=[go.Bar(
            x=age_avg['asset_name'],
            y=age_avg[avg_column],
            hovertext=hover_text_age,
            hoverinfo='text+y'
        )])
        fig.update_layout(
            title=age,
            title_x=0.5,
            plot_bgcolor='#222222',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#E0E0E0',
            height=300,
            yaxis_title=yaxis_title
        )
        figs_age.append(fig)
        tables_age.append(age_avg)
    
    # Debug the contents of tables_age
    print("Contents of tables_age:")
    for i, table in enumerate(tables_age):
        print(f"tables_age[{i}] type: {type(table)}")
        if isinstance(table, pd.DataFrame):
            print(f"tables_age[{i}] columns: {table.columns.tolist()}")
        else:
            print(f"tables_age[{i}] value: {table}")

    print(f"Returning data for {selected_manufacturer} with {len(filtered_data)} assets")
    return [
        fig_total,
        total_avg.to_dict('records'),
        [{"name": i, "id": i} for i in total_avg.columns],
        figs_age[0],
        tables_age[0].to_dict('records'),
        [{"name": i, "id": i} for i in tables_age[0].columns],
        figs_age[1],
        tables_age[1].to_dict('records'),
        [{"name": i, "id": i} for i in tables_age[1].columns],
        figs_age[2],
        tables_age[2].to_dict('records'),
        [{"name": i, "id": i} for i in tables_age[2].columns],
        message
    ]

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

