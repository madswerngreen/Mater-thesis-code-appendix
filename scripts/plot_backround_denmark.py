import geopandas as gpd

file = '../Data/Geospatial_info/Zoneslevel3_GMM4_with_CRS.gpkg'
zones = gpd.read_file(file)
zones = zones.loc[zones['zoneid'].astype(int) < 900000].copy()

# dissolve all into a single multipart geometry
dk = zones.dissolve()          # no by= argument → everything becomes one feature

# plot
ax = dk.plot(
    color='lightgray',
    edgecolor='black',
    linewidth=0.6,
    alpha=0.6,
    figsize=(6,10)
)

ax.set_aspect('equal')
