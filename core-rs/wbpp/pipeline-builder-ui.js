// Select this Pipeline Builder to edit DWT settings before WBPP processing.
{
   function editDwtSettings( config, numericSettings, saveSettings )
   {
      let dialog = new Dialog;
      dialog.windowTitle = "DWT-Drizzle (Rust) Settings";

      let tabs = new TabBox( dialog );
      let numericFields = [];
      let modes = [ "max", "band", "ctn" ];
      let mode;
      let seeds;
      let drizzle;
      let autoReject;
      let fieldCorrection;
      let pages = [ "Stack", "PWS", "R field" ];
      for ( let pageIndex = 0; pageIndex < pages.length; ++pageIndex )
      {
         let title = pages[ pageIndex ];
         let page = new Control( tabs );
         page.sizer = new VerticalSizer;
         page.sizer.margin = 8;
         page.sizer.spacing = 4;
         let controls = [];
         if ( title == "R field" )
         {
            mode = new ComboBox( page );
            mode.addItem( "Local peak" );
            mode.addItem( "Band-pass" );
            mode.addItem( "Contrast / noise" );
            mode.currentItem = modes.indexOf( config.rFieldMode );
            controls.push( { title: "R field mode", control: mode } );
         }
         for ( let field of numericSettings.filter( field => field.page == title ) )
         {
            let edit = new Edit( page );
            let value = config[ field.key ];
            let initialText = String( Number( value.toPrecision( 9 ) ) );
            edit.text = initialText;
            edit.setScaledFixedWidth( 96 );
            edit.toolTip = field.hint;
            numericFields.push( { field: field, edit: edit, pageIndex: pageIndex, initialText: initialText } );
            controls.push( { title: field.label, control: edit } );
         }
         page.sizer.add( makeFieldGrid( page, controls ) );
         if ( title == "Stack" )
         {
            drizzle = new CheckBox( page );
            drizzle.text = "Enable Drizzle scaling";
            drizzle.toolTip = "When cleared, registration and integration use a 1× grid with pixfrac 1.";
            drizzle.checked = config.drizzleEnabled;
            let drizzleEdits = numericFields.filter( entry => [ "drizzleScale", "drizzlePixfrac" ].includes( entry.field.key ) );
            let updateDrizzleEdits = checked => drizzleEdits.forEach( entry => entry.edit.enabled = checked );
            drizzle.onCheck = updateDrizzleEdits;
            updateDrizzleEdits( drizzle.checked );

            seeds = new CheckBox( page );
            seeds.text = "Use WBPP XDRZ seeds";
            seeds.toolTip = "Use existing WBPP sidecars as initial transforms for Rust registration.";
            seeds.checked = config.useXdrzSeeds;
            let switches = new HorizontalSizer;
            switches.spacing = 12;
            switches.add( drizzle );
            switches.add( seeds );
            switches.addStretch();
            page.sizer.add( switches );
            autoReject = new CheckBox( page );
            autoReject.text = "Reject isolated low-throughput frames";
            autoReject.toolTip = "Compare matched stars with neighboring exposures. Requires observation timestamps.";
            autoReject.checked = config.autoRejectFrames;
            fieldCorrection = new CheckBox( page );
            fieldCorrection.text = "Refine field geometry";
            fieldCorrection.toolTip = "Apply a smooth correction when held-out stars confirm lower registration residuals.";
            fieldCorrection.checked = config.fieldCorrection;
            let quality = new HorizontalSizer;
            quality.spacing = 12;
            quality.add( autoReject );
            quality.add( fieldCorrection );
            quality.addStretch();
            page.sizer.add( quality );
         }
         page.sizer.addStretch();
         tabs.addPage( page, title );
      }

      let status = new Label( dialog );
      status.text = " ";
      status.wordWrapping = true;
      let confirm = new PushButton( dialog );
      confirm.text = "Save and continue";
      confirm.defaultButton = true;
      confirm.onClick = function()
      {
         let settings = {};
         for ( let entry of numericFields )
         {
            let text = entry.edit.text.trim();
            let value = text == entry.initialText ? config[ entry.field.key ] : Number( text );
            if ( text.length == 0 || !Number.isFinite( Math.fround( value ) ) || !entry.field.valid( value ) )
            {
               if ( !entry.edit.enabled )
                  continue;
               status.text = entry.field.label + ". " + entry.field.hint;
               tabs.currentPageIndex = entry.pageIndex;
               entry.edit.hasFocus = true;
               return;
            }
            settings[ entry.field.key ] = value;
         }
         settings.rFieldMode = modes[ mode.currentItem ];
         settings.useXdrzSeeds = seeds.checked;
         settings.drizzleEnabled = drizzle.checked;
         settings.autoRejectFrames = autoReject.checked;
         settings.fieldCorrection = fieldCorrection.checked;
         try
         {
            saveSettings( settings );
         }
         catch ( error )
         {
            status.text = "Could not save project settings. See the Process Console.";
            console.criticalln( "** Error saving DWT project settings: " + ( error.message || String( error ) ) );
            console.flush();
            return;
         }
         Object.assign( config, settings );
         dialog.ok();
      };

      let cancel = new PushButton( dialog );
      cancel.text = "Cancel run";
      cancel.onClick = function() { dialog.cancel(); };
      let buttons = new HorizontalSizer;
      buttons.spacing = 8;
      buttons.addStretch();
      buttons.add( confirm );
      buttons.add( cancel );

      dialog.sizer = new VerticalSizer;
      dialog.sizer.margin = 8;
      dialog.sizer.spacing = 4;
      dialog.sizer.add( tabs, 100 );
      dialog.sizer.add( status );
      dialog.sizer.add( buttons );
      dialog.adjustToContents();
      return dialog.execute() == StdDialogCode.Ok;
   }

   function makeFieldGrid( page, controls )
   {
      let grid = new HorizontalSizer;
      grid.spacing = 16;
      for ( let column = 0; column < 2; ++column )
      {
         let rows = new VerticalSizer;
         rows.spacing = 4;
         let labels = [];
         for ( let i = column; i < controls.length; i += 2 )
         {
            let entry = controls[ i ];
            let label = new Label( page );
            label.text = entry.title;
            label.textAlignment = TextAlignment.Left | TextAlignment.VertCenter;
            labels.push( label );
            let row = new HorizontalSizer;
            row.spacing = 6;
            row.add( label );
            row.add( entry.control );
            row.addStretch();
            rows.add( row );
         }
         let width = Math.max( ...labels.map( label => label.font.width( label.text ) ) );
         for ( let label of labels )
            label.setFixedWidth( width );
         rows.addStretch();
         grid.add( rows, 100 );
      }
      return grid;
   }

   let scriptPath = engine.pipelineBuilderScriptFile;
   let directory = File.extractDrive( scriptPath ) + File.extractDirectory( scriptPath );
   eval( File.readTextFile( directory + "/pipeline-builder.js" ) );
}
