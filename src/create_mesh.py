import subprocess
import geojson
import icepack

def fetch_outline(name='pine-island'):
    """
    Fetches outline data for a given name.

    :param name: Name of the outline (default is 'pine-island').
    :return: GeoJSON outline data.
    """
    try:
        if "." in name:
            print("Reading local outline")
            outline_filename = name
        else:
            print("Reading outline from icepack database")
            outline_filename = icepack.datasets.fetch_outline(name)
        with open(outline_filename, "r") as outline_file:
            outline = geojson.load(outline_file)
        return outline
    except Exception as e:
        print(f"Error fetching outline: {e}")
        return None

def create_geo_file(outline, name='pig', lcar = 10e3):
    """
    Creates a geo file from the given outline.

    :param outline: GeoJSON outline data.
    :param name: Name for the geo file (default is 'pig').
    """
    if "." in name:
        name = name.split('.')[0]
    geo_name = name + '.geo'
    try:
        geometry = icepack.meshing.collection_to_geo(outline,  lcar=lcar)
        with open(geo_name, "w") as geo_file:
            geo_file.write(geometry.get_code())
    except Exception as e:
        print(f"Error creating geo file: {e}")

def create_mesh(outline, name='pig', lcar = 10e3, **kwargs):
    """
    Creates a mesh using Gmsh.

    :param outline: GeoJSON outline data.
    :param name: Name for the mesh (default is 'pig').
    :param kwargs: Additional keyword arguments for Gmsh.
    """
    create_geo_file(outline, name, lcar)
    mesh_name = name + '.msh'
    geo_name = name + '.geo'
    try:
        command = f"gmsh -2 -format msh2 -v 2 -o {mesh_name} {geo_name}"
        subprocess.run(command.split(), **kwargs)
    except Exception as e:
        print(f"Error creating mesh: {e}")

def check_available_outlines():
    """
    Checks and returns the available glacier names in the icepack datasets.

    :return: List of available glacier names.
    """
    return icepack.datasets.get_glacier_names()

# Example usage:
if __name__ == "__main__":
    outline_data = fetch_outline()
    if outline_data:
        create_mesh(outline_data)