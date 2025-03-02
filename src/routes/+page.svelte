<script lang="ts">
  import { onMount } from 'svelte';
  import mapboxgl from 'mapbox-gl';
  import 'mapbox-gl/dist/mapbox-gl.css';

  let mapContainer: HTMLDivElement;

  // Station data GeoJSON with type annotations
  const stationData: GeoJSON.FeatureCollection = {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [-79.3128, 43.6864]
        },
        "properties": {
          "station": "WOODBINE STATION",
          "hour": 0,
          "avg_delay": 2.15,
          "incident_count": 33
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [-79.3128, 43.6864]
        },
        "properties": {
          "station": "WOODBINE STATION",
          "hour": 1,
          "avg_delay": 1.05,
          "incident_count": 2233
        }
      },
      // ... Add all your other features here ...
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [-79.3231, 43.6841]
        },
        "properties": {
          "station": "COXWELL STATION",
          "hour": 23,
          "avg_delay": 3.17,
          "incident_count": 499999
        }
      }
    ]
  };

  // Add this after stationData
  const subwayLines: GeoJSON.FeatureCollection = {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "properties": { "line": "1", "color": "#FED103" },
        "geometry": {
          "type": "LineString",
          "coordinates": [
            [-79.5126, 43.6387], // Kipling
            [-79.5080, 43.6436], // Islington
            [-79.4950, 43.6481], // Royal York
            [-79.4849, 43.6508], // Old Mill
            [-79.4676, 43.6510], // Jane
            [-79.4536, 43.6516], // Runnymede
            [-79.4399, 43.6549], // High Park
            [-79.4260, 43.6549], // Keele
            [-79.4133, 43.6561], // Dundas West
            [-79.4020, 43.6563], // Lansdowne
            [-79.3936, 43.6609], // Dufferin
            [-79.3827, 43.6624], // Ossington
            [-79.3722, 43.6641], // Christie
            [-79.3641, 43.6648], // Bathurst
            [-79.3523, 43.6651], // Spadina
            [-79.3987, 43.6685], // St George
            [-79.3932, 43.6677], // Museum
            [-79.3903, 43.6606], // Queen's Park
            [-79.3886, 43.6547], // St Patrick
            [-79.3873, 43.6505], // Osgoode
            [-79.3839, 43.6474], // St Andrew
            [-79.3792, 43.6453], // Union
            [-79.3775, 43.6700], // Bloor-Yonge
            [-79.3798, 43.6821], // Rosedale
            [-79.3806, 43.6881], // Summerhill
            [-79.3833, 43.6878], // St Clair
            [-79.3857, 43.7153], // Davisville
            [-79.3973, 43.7332], // Eglinton
            [-79.4025, 43.7410], // Lawrence
            [-79.4123, 43.7531], // York Mills
            [-79.4116, 43.7667], // Sheppard-Yonge
            [-79.4141, 43.7681], // North York Centre
            [-79.4150, 43.7884]  // Finch
          ]
        }
      },
      {
        "type": "Feature",
        "properties": { "line": "2", "color": "#00923F" },
        "geometry": {
          "type": "LineString",
          "coordinates": [
            [-79.5238, 43.6366], // Kipling
            [-79.5157, 43.6374], // Islington
            [-79.4972, 43.6389], // Royal York
            [-79.4762, 43.6461], // Kennedy
            [-79.4619, 43.6544], // Main Street
            [-79.4231, 43.6677], // Woodbine
            [-79.3231, 43.6841], // Coxwell
            [-79.3775, 43.6700], // Bloor-Yonge
            [-79.3987, 43.6685], // St George
            [-79.4110, 43.6676], // Spadina
            [-79.4273, 43.6666], // Bathurst
            [-79.4388, 43.6664], // Christie
            [-79.4669, 43.6659]  // Ossington
          ]
        }
      },
      {
        "type": "Feature",
        "properties": { "line": "4", "color": "#8C288C" },
        "geometry": {
          "type": "LineString",
          "coordinates": [
            [-79.3469, 43.7755], // Don Mills
            [-79.4116, 43.7667], // Sheppard-Yonge
            [-79.3873, 43.7697], // Bayview
            [-79.3645, 43.7717]  // Leslie
          ]
        }
      }
    ]
  };

  onMount(() => {
    const map = new mapboxgl.Map({
      container: mapContainer,
      style: 'mapbox://styles/mapbox/dark-v11', // Changed to dark style for better visualization
      center: [-79.3128, 43.6864], // Negative longitude
      zoom: 13,
      pitch: 0, // Removed tilt for better data visibility
      antialias: true,
      accessToken: 'pk.eyJ1IjoicGVhY2hlc2dvYmJsciIsImEiOiJjbTdyMzQxMXQxNGNmMmpwdXJrYWd0c3M4In0.rkr_jHMuuHPlbCgGAk_q8w'
    });

    map.on('load', () => {
      // Add subway lines source
      map.addSource('subway-lines', {
        type: 'geojson',
        data: subwayLines
      });

      // Add subway lines layer
      map.addLayer({
        'id': 'subway-routes',
        'type': 'line',
        'source': 'subway-lines',
        'paint': {
          'line-color': ['get', 'color'],
          'line-width': 3
        }
      });

      // Add existing stations source and layer
      map.addSource('stations', {
        type: 'geojson',
        data: stationData
      });

      // Add a layer for the stations
      map.addLayer({
        'id': 'station-delays',
        'type': 'circle',
        'source': 'stations',
        'paint': {
          // Circle radius based on incident count - increased size range
          'circle-radius': [
            'interpolate',
            ['linear'],
            ['get', 'incident_count'],
            0, 15,    // Minimum size increased to 15px
            82, 50    // Maximum size increased to 50px
          ],
          // Circle color based on average delay
          'circle-color': [
            'interpolate',
            ['linear'],
            ['get', 'avg_delay'],
            0, '#00ff00',
            9, '#ff0000'
          ],
          'circle-opacity': 0.7,
          'circle-stroke-width': 1,
          'circle-stroke-color': '#ffffff'
        }
      });

      // Add popup on hover
      const popup = new mapboxgl.Popup({
        closeButton: false,
        closeOnClick: false
      });

      map.on('mouseenter', 'station-delays', (e: mapboxgl.MapMouseEvent & { features?: mapboxgl.MapboxGeoJSONFeature[] }) => {
        if (!e.features?.length) return;
        
        map.getCanvas().style.cursor = 'pointer';
        
        const feature = e.features[0];
        const point = feature.geometry as GeoJSON.Point;
        const coordinates: [number, number] = [point.coordinates[0], point.coordinates[1]];
        
        const properties = feature.properties;
        if (!properties) return;
        
        const description = `
          <strong>${properties.station}</strong><br>
          Hour: ${properties.hour}:00<br>
          Average Delay: ${properties.avg_delay} minutes<br>
          Incident Count: ${properties.incident_count}
        `;
        
        popup
          .setLngLat(coordinates)
          .setHTML(description)
          .addTo(map);
      });

      map.on('mouseleave', 'station-delays', () => {
        map.getCanvas().style.cursor = '';
        popup.remove();
      });
    });

    return () => map.remove();
  });
</script>

<div class="container">
  <h1 class="text-2xl mb-4">Toronto Transit Delays by Station</h1>
  <div class="legend mb-4">
    <p class="text-sm mb-2">Circle size represents incident count</p>
    <p class="text-sm">Circle color represents average delay (green = low, red = high)</p>
  </div>
  <div bind:this={mapContainer} class="map-container"></div>
</div>

<style>
  .container {
    padding: 1rem;
  }

  .map-container {
    width: 100%;
    height: 600px;
    border-radius: 4px;
    border: 1px solid #ccc;
  }

  .legend {
    background: rgba(255, 255, 255, 0.9);
    padding: 0.5rem;
    border-radius: 4px;
  }

  :global(.mapboxgl-map) {
    width: 100%;
    height: 100%;
  }

  :global(.mapboxgl-popup) {
    max-width: 300px;
  }

  :global(.mapboxgl-popup-content) {
    padding: 1rem;
    font-size: 14px;
  }
</style>