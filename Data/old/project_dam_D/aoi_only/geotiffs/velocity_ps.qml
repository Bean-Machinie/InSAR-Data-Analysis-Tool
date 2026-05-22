<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer opacity="1" alphaBand="-1" band="1" type="singlebandpseudocolor" classificationMin="-284.940002" classificationMax="284.940002">
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" clip="0">
          <item value="-284.940002" label="-284.94 mm/year" color="#2166ac" alpha="255"/>
          <item value="0.000000" label="0.00 mm/year" color="#f7f7f7" alpha="255"/>
          <item value="284.940002" label="284.94 mm/year" color="#b2182b" alpha="255"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation colorizeOn="0" grayscaleMode="0" saturation="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <legend type="default-vector"/>
  <blendMode>0</blendMode>
  <layerOpacity>1</layerOpacity>
  <customproperties>
    <Option type="Map">
      <Option name="insar_units" type="QString" value="mm/year"/>
      <Option name="insar_style_note" type="QString" value="AOI PS velocity: negative is blue, positive is red, zero is white"/>
    </Option>
  </customproperties>
</qgis>
