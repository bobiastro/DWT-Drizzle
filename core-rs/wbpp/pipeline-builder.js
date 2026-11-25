// WBPP injection point for DWT-Drizzle.
//
// Select this file as Pipeline builder in WBPP:
//   Pipeline > Pipeline scripting > Pipeline builder
//
// The standard WBPP light pipeline is assembled first. The DWT operation is
// queued before standard integration. Rust can register the calibrated frames.

engine.pipelineManager.buildPipelineForLight();

let dwtOperationIndex = findDwtInsertionIndex();
let dwtRunConfig;
let dwtOperationBlock = new BPPOperationBlock( "DWT-Drizzle (Rust)", undefined, true );
let dwtSpace = {
   label: dwtOperationBlock.name,
   size: estimateDwtDiskSpace( readDwtConfiguration( engine, false, false ) )
};
dwtOperationBlock.spaceRequired = () => dwtSpace.size;
if ( engine.WS )
   engine.WS.dwtDrizzle = dwtSpace;
dwtOperationBlock.envForScript = function()
{
   return { name: this.name, status: this.status };
};
dwtOperationBlock._run = function( env, requestInterruption )
{
   console.noteln( "* DWT-Drizzle Pipeline builder: <raw>" + engine.pipelineBuilderScriptFile + "</raw>" );
   console.flush();
   let dwtDirectory = engine.outputDirectory + "/DWT";
   let cfg = dwtRunConfig || readDwtConfiguration( engine, false );
   let excludedFrames = new Set( cfg.excludeFrames.map( dwtPathKey ) );

   let groups = dwtLightGroups();
   let completed = 0;
   for ( let g = 0; g < groups.length; ++g )
   {
      if ( dwtInterruptionRequested() )
         return OperationBlockStatus.CANCELED;
      let group = groups[ g ];
      if ( group.fastIntegrationData && group.fastIntegrationData.enabled )
      {
         console.warningln( "** Warning: DWT-Drizzle skips FAST Integration group " + group.folderName( false ) + "." );
         continue;
      }
      let activeFrames = group.activeFrames().filter( function( frame )
      {
         let source = dwtSourceForFrame( frame );
         if ( source == undefined )
            return false;
         let input = frame.fileItem.filePath;
         if ( dwtFrameIsExcluded( frame, source, excludedFrames ) )
         {
            console.noteln( "* DWT-Drizzle excluded frame: " + input );
            return false;
         }
         return true;
      } );
      if ( activeFrames.length == 0 )
         continue;
      if ( group.isCFA && group.associatedRGBchannel == undefined )
      {
         console.warningln( "** Warning: DWT-Drizzle skips undebayered CFA group " + group.folderName( false ) + "." );
         continue;
      }
      if ( cfg.frameLimit > 0 && activeFrames.length > cfg.frameLimit )
      {
         console.noteln( "* DWT-Drizzle frame limit: using the first " + cfg.frameLimit + " of " + activeFrames.length + " active frames" );
         activeFrames = activeFrames.slice( 0, cfg.frameLimit );
      }
      let firstSource = dwtSourceForFrame( activeFrames[ 0 ] );
      let firstWindow = ImageWindow.open( firstSource )[ 0 ];
      if ( firstWindow == null || firstWindow.isNull )
      {
         console.warningln( "** Warning: cannot open calibrated frame: " + firstSource );
         continue;
      }
      let width = firstWindow.mainView.image.width;
      let height = firstWindow.mainView.image.height;
      firstWindow.forceClose();
      let groupName = sanitizeDwtName( group.folderName( false ) );
      dwtOperationBlock.groupDescription = groupName + " | Exporting calibrated frames";
      console.noteln( "* DWT-Drizzle exporting " + activeFrames.length + " calibrated frames for " + groupName );
      console.flush();
      let groupHash = new CryptographicHash( CryptographicHash.SHA256 );
      let groupSuffix = String( groupHash.hash( group.id ).toHex() ).slice( 0, 12 );
      let groupDirectory = dwtDirectory + "/runs/" + groupName + "~" + groupSuffix;
      ensureDwtDirectory( groupDirectory );
      let prepareRequest = makeDwtRequest( engine, groupDirectory, width, height, cfg );
      for ( let f = 0; f < activeFrames.length; ++f )
      {
         dwtOperationBlock.groupDescription = groupName + " | Exporting calibrated frame "
            + ( f + 1 ) + "/" + activeFrames.length;
         if ( dwtInterruptionRequested() )
            return OperationBlockStatus.CANCELED;
         let frame = activeFrames[ f ];
         let source = dwtSourceForFrame( frame );
         let stem = sanitizeDwtName( File.extractName( source ) );
         let rasterFile = groupDirectory + "/" + f + "-" + stem + ".f32le";
         if ( !writeCalibratedFloat32( source, rasterFile, width, height ) )
         {
            console.warningln( "** Warning: cannot export frame as a matching single-channel raster: " + source );
            continue;
         }
         let descriptor = frame.descriptor || {};
         let processed = frame.fileItem.processed[ frame.associatedID ] || {};
         let solver = frame.fileItem.solverParams || {};
         let timestamp = solver.timestamp || solver.observationTimestamp;
         let xdrz = cfg.useXdrzSeeds && frame.drizzleFile ? frame.drizzleFile : null;
         prepareRequest.frames.push( {
            enabled: true,
            source: source,
            registered: processed[ BPP.FrameProcessingStep.REGISTRATION ] || null,
            xdrz: xdrz,
            has_xdrz: xdrz != null,
            fwhm: isFinite( descriptor.FWHM ) ? descriptor.FWHM : null,
            noise: isFinite( descriptor.noise ) ? descriptor.noise : null,
            signal: isFinite( descriptor.PSFSignalWeight ) ? descriptor.PSFSignalWeight : null,
            exposure: isFinite( frame.exposureTime ) ? frame.exposureTime : null,
            observation_timestamp_ms: Number.isSafeInteger( timestamp ) && timestamp > 0 ? timestamp : null,
            raster: rasterFile,
            background: isFinite( descriptor.median ) ? descriptor.median : null
         } );
      }
      if ( dwtInterruptionRequested() )
         return OperationBlockStatus.CANCELED;
      let requestFile = groupDirectory + "/wbpp-request.json";
      File.writeTextFile( requestFile, JSON.stringify( prepareRequest, null, 2 ) );
      cfg.requestFile = requestFile;
      if ( prepareRequest.frames.length == 0 )
      {
         console.warningln( "** Warning: no calibrated frames were exported for DWT-Drizzle group " + groupName + "." );
         continue;
      }
      let status = executeDwtGroup( env, requestInterruption, cfg, requestFile, groupDirectory, groupName );
      if ( status == OperationBlockStatus.FAILED )
         this.statusMessage = "DWT-Drizzle failed for " + groupName;
      if ( status != OperationBlockStatus.DONE )
         return status;
      completed++;
   }

   if ( completed == 0 )
   {
      console.warningln( "** Warning: no DWT-Drizzle light groups were processed." );
      env.params.processLogger.addWarning( "No DWT-Drizzle light groups were processed." );
      this.hasWarnings = true;
   }
   this.statusMessage = completed + " light group(s) integrated";
   console.noteln( "* DWT-Drizzle handoff complete: " + completed + " group(s)." );
   console.flush();
   return OperationBlockStatus.DONE;
};
engine.operationQueue.operations.splice( dwtOperationIndex, 0, {
   operation: dwtOperationBlock,
   params: {
      processLogger: engine.processLogger
   }
} );
let firstChangedIndex = dwtOperationIndex;
if ( typeof editDwtSettings == "function" && !engine.automationMode )
{
   let settingsOperation = new BPPOperationBlock( "DWT-Drizzle Settings", undefined, true );
   settingsOperation._run = function( env, requestInterruption )
   {
      try
      {
         dwtRunConfig = undefined;
         let cfg = readDwtConfiguration( engine, true );
         if ( !editDwtSettings( cfg, dwtNumericSettings(), settings => saveDwtProjectSettings( engine, settings ) ) )
         {
            this.statusMessage = "DWT-Drizzle settings canceled";
            requestInterruption();
            return OperationBlockStatus.CANCELED;
         }
         dwtRunConfig = cfg;
         dwtSpace.size = estimateDwtDiskSpace( cfg );
         this.statusMessage = "Project settings saved";
         return OperationBlockStatus.DONE;
      }
      catch ( error )
      {
         let message = error.message || String( error );
         this.statusMessage = message;
         console.criticalln( "** Error: " + message );
         env.params.processLogger.addError( message );
         requestInterruption();
         return OperationBlockStatus.FAILED;
      }
   };
   engine.operationQueue.operations.unshift( { operation: settingsOperation, params: { processLogger: engine.processLogger } } );
   firstChangedIndex = 0;
}
for ( let i = firstChangedIndex; i < engine.operationQueue.operations.length; ++i )
   engine.operationQueue.operations[ i ].operation.__index__ = i;

function dwtLightGroups()
{
   return engine.groupsManager.groupsForMode( BPP.GroupingMode.POST ).filter( function( group )
   {
      return group.isActive && group.imageType == ImageType.Light
         && group.associatedRGBchannel != BPP.AssociatedChannel.COMBINED_RGB;
   } );
}

function dwtFrameIsExcluded( frame, source, excludedFrames )
{
   return excludedFrames.has( dwtPathKey( frame.fileItem.filePath ) )
      || ( source != undefined && excludedFrames.has( dwtPathKey( source ) ) );
}

function estimateDwtDiskSpace( cfg )
{
   let scale = cfg.drizzleEnabled ? Math.fround( cfg.drizzleScale ) : 1;
   let excludedFrames = new Set( cfg.excludeFrames.map( dwtPathKey ) );
   let bytes = 0;
   for ( let group of dwtLightGroups() )
   {
      if ( ( group.fastIntegrationData && group.fastIntegrationData.enabled )
         || ( group.isCFA && group.associatedRGBchannel == undefined ) )
         continue;
      let size = group.size;
      if ( !size || !( size.width > 0 && size.height > 0 ) )
         continue;
      let frames = group.activeFrames();
      if ( excludedFrames.size > 0 )
         frames = frames.filter( frame => !dwtFrameIsExcluded( frame, dwtSourceForFrame( frame ), excludedFrames ) );
      let count = cfg.frameLimit > 0 ? Math.min( frames.length, cfg.frameLimit ) : frames.length;
      if ( count == 0 )
         continue;

      let inputBytes = size.width * size.height * 4 * count;
      // Use the nominal canvas with a one-pixel margin on each side.
      let outputPixels = Math.ceil( ( size.width + 2 ) * scale ) * Math.ceil( ( size.height + 2 ) * scale );
      // Three 4-byte output rasters and one float32 XISF master.
      bytes += inputBytes + outputPixels * 16;
   }
   return Math.min( bytes, Number.MAX_SAFE_INTEGER );
}

function makeDwtRequest( engine, outputDirectory, width, height, config )
{
   return {
      schema: "dwt-wbpp-command/1",
      source: "PixInsight/WBPP",
      calibrated_directory: engine.outputDirectory + "/calibrated",
      registered_directory: engine.outputDirectory + "/registered",
      output_directory: outputDirectory,
      frames: [],
      raster: {
         directory: outputDirectory,
         file_extension: "f32le",
         width: width,
         height: height,
         dtype: "float32-le"
      },
      options: {
         clarity_exponent: config.clarityExponent,
         snr_exponent: config.snrExponent,
         snr_clip: config.snrClip,
         detail_snr: config.detailSnr,
         flux_power: config.fluxPower,
         sharp_weight: config.sharpWeight,
         rejection_sigma: config.rejectionSigma,
         rejection_fraction: config.rejectionFraction,
         drizzle_scale: config.drizzleEnabled ? config.drizzleScale : 1.0,
         drizzle_pixfrac: config.drizzleEnabled ? config.drizzlePixfrac : 1.0,
         registration_stars: config.registrationStars,
         fwhm_max: config.fwhmMax,
         auto_reject_frames: config.autoRejectFrames,
         field_correction: config.fieldCorrection,
         stack_workers: config.stackWorkers,
         allow_missing_xdrz: true,
         env_window_multiplier: config.envWindowMultiplier,
         env_epsilon: config.envEpsilon,
         env_power: config.envPower,
         r_floor: config.rFieldFloor,
         r_blur: config.rFieldBlur,
         r_mode: config.rFieldMode,
         r_window: config.rFieldWindow,
         r_contrast: config.rFieldContrast,
         ref_frames: config.referenceFrames
      }
   };
}

function executeDwtGroup( env, requestInterruption, cfg, requestFile, groupDirectory, groupName )
{
   console.noteln( "* DWT-Drizzle: " + groupName );
   if ( !File.exists( cfg.executable ) )
   {
      console.criticalln( "** Error: Rust backend executable not found: " + cfg.executable );
      env.params.processLogger.addError( "Rust backend executable not found: " + cfg.executable );
      requestInterruption();
      return OperationBlockStatus.FAILED;
   }
   let resultFile = groupDirectory + "/dwt-result.json";
   let result = runExternalProcess( cfg.executable, [ "wbpp-pws", requestFile, resultFile ], function( progress )
   {
      dwtOperationBlock.groupDescription = groupName + " | " + progress;
      console.noteln( "* DWT-Drizzle " + progress );
      console.flush();
   } );
   if ( result.canceled )
      return OperationBlockStatus.CANCELED;
   if ( !result.ok )
   {
      console.criticalln( "** Error: DWT-Drizzle failed for " + groupName + "." );
      if ( result.stdout.length > 0 )
         console.writeln( result.stdout );
      if ( result.stderr.length > 0 )
         console.criticalln( result.stderr );
      env.params.processLogger.addError( "DWT-Drizzle failed for " + groupName + "." );
      requestInterruption();
      return OperationBlockStatus.FAILED;
   }
   if ( result.stdout.length > 0 )
      console.writeln( result.stdout );
   let resultJson;
   try
   {
      resultJson = JSON.parse( File.readTextFile( resultFile ) );
   }
   catch ( error )
   {
      console.criticalln( "** Error: Rust result metadata could not be read for " + groupName + "." );
      env.params.processLogger.addError( "Rust result metadata could not be read for " + groupName + "." );
      requestInterruption();
      return OperationBlockStatus.FAILED;
   }
   if ( resultJson.status != "ok" || !Array.isArray( resultJson.raster_shape )
      || resultJson.raster_shape.length != 2
      || !resultJson.raster_shape.every( value => Number.isInteger( value ) && value > 0 ) )
   {
      env.params.processLogger.addError( "Invalid Rust result metadata for " + groupName + "." );
      requestInterruption();
      return OperationBlockStatus.FAILED;
   }
   let outputWidth = resultJson.raster_shape[ 1 ];
   let outputHeight = resultJson.raster_shape[ 0 ];
   let integrated = resultJson.integrated_frames ?? resultJson.registered_frames;
   let frameSummary = "DWT-Drizzle integrated " + integrated + "/" + resultJson.frame_count
      + " frames for " + groupName + " (registered " + resultJson.registered_frames
      + ", rejected by throughput " + ( resultJson.quality_rejected_frames ?? 0 )
      + ", sharp reference " + resultJson.sharp_reference_frames + ")";
   env.params.processLogger.addMessage( frameSummary );
   if ( resultJson.field_corrected_frames > 0 )
   {
      let correctionSummary = "DWT-Drizzle field correction in " + resultJson.field_corrected_frames
         + " frame(s), median validation residual " + resultJson.field_median_residual_before.toFixed( 2 )
         + " -> " + resultJson.field_median_residual_after.toFixed( 2 ) + " px";
      console.noteln( "* " + correctionSummary );
      env.params.processLogger.addMessage( correctionSummary );
   }
   for ( let warning of ( resultJson.warnings || [] ) )
   {
      console.warningln( "** Warning: " + warning );
      env.params.processLogger.addWarning( warning );
   }
   if ( dwtInterruptionRequested() )
      return OperationBlockStatus.CANCELED;
   let outputFile = String( resultJson.output_raster );
   let outputWindow = readFloat32Window( outputFile, outputWidth, outputHeight );
   if ( outputWindow == null )
   {
      console.criticalln( "** Error: Rust PWS output could not be loaded for " + groupName + "." );
      env.params.processLogger.addError( "Rust PWS output could not be loaded for " + groupName + "." );
      requestInterruption();
      return OperationBlockStatus.FAILED;
   }
   if ( dwtInterruptionRequested() )
   {
      outputWindow.forceClose();
      return OperationBlockStatus.CANCELED;
   }
   outputWindow.mainView.id = "DWT_" + groupName.replace( /[^A-Za-z0-9_]/g, "_" );
   let outputPath = groupDirectory + "/master_dwt_" + groupName + ".xisf";
   dwtOperationBlock.groupDescription = groupName + " | Saving master XISF";
   console.noteln( "* DWT-Drizzle saving master XISF for " + groupName );
   console.flush();
   let saved = outputWindow.saveAs( outputPath, false, false, true, false );
   outputWindow.forceClose();
   if ( !saved )
   {
      env.params.processLogger.addError( "DWT-Drizzle could not save " + outputPath );
      requestInterruption();
      return OperationBlockStatus.FAILED;
   }
   console.noteln( "  output: <raw>" + outputPath + "</raw>" );
   env.params.processLogger.addSuccess( "DWT-Drizzle PWS generated", outputPath );
   return OperationBlockStatus.DONE;
}

function readDwtConfiguration( engine, strictProject, reportWarnings = true )
{
   let scriptPath = engine.pipelineBuilderScriptFile;
   let scriptDirectory = File.extractDrive( scriptPath ) + File.extractDirectory( scriptPath );
   let prefix = scriptDirectory.length > 0 ? scriptDirectory + "/" : "";
   let executableName = CoreApplication.platform == "Windows" ? "dwt-backend.exe" : "dwt-backend";
   let config = {
      executable: prefix + "bin/" + executableName,
      requestFile: "",
      useXdrzSeeds: false,
      excludeFrames: [],
      autoRejectFrames: true,
      fieldCorrection: true,
      drizzleEnabled: true,
      rFieldMode: "max"
   };
   for ( let field of dwtNumericSettings() )
      config[ field.key ] = field.initial;
   applyDwtConfiguration( config, readDwtConfigObject( prefix + "dwt-backend.json", false, reportWarnings ) );
   if ( engine.outputDirectory )
      applyDwtConfiguration( config,
         readDwtConfigObject( engine.outputDirectory + "/DWT/dwt-backend.json", strictProject, reportWarnings ) );
   return config;
}

function saveDwtProjectSettings( engine, settings )
{
   let directory = engine.outputDirectory + "/DWT";
   let path = directory + "/dwt-backend.json";
   let project = readDwtConfigObject( path, true );
   ensureDwtDirectory( directory );
   for ( let key of Object.keys( settings ) )
      project[ key ] = settings[ key ];
   WBPPUtils.writeTextFileAtomically( path, JSON.stringify( project, null, 2 ) + "\n" );
}

function ensureDwtDirectory( path )
{
   if ( File.directoryExists( path ) )
      return;
   try
   {
      File.createDirectory( path, true );
   }
   catch ( error )
   {
      if ( !File.directoryExists( path ) )
         throw error;
   }
}

function readDwtConfigObject( configFile, strict, reportWarnings = true )
{
   if ( !File.exists( configFile ) )
      return {};
   try
   {
      let parsed = JSON.parse( File.readTextFile( configFile ).replace( /^\uFEFF/, "" ) );
      if ( parsed === null || typeof parsed != "object" || Array.isArray( parsed ) )
         throw new Error( "Expected a JSON object" );
      return parsed;
   }
   catch ( error )
   {
      let message = "Invalid DWT configuration in " + configFile + " (" + ( error.message || String( error ) ) + ")";
      if ( strict )
         throw new Error( message );
      if ( reportWarnings )
         console.warningln( "** Warning: " + message );
      return {};
   }
}

function applyDwtConfiguration( config, parsed )
{
   if ( parsed.executable )
      config.executable = parsed.executable;
   if ( parsed.requestFile )
      config.requestFile = parsed.requestFile;
   if ( typeof parsed.useXdrzSeeds == "boolean" )
      config.useXdrzSeeds = parsed.useXdrzSeeds;
   if ( typeof parsed.autoRejectFrames == "boolean" )
      config.autoRejectFrames = parsed.autoRejectFrames;
   if ( typeof parsed.fieldCorrection == "boolean" )
      config.fieldCorrection = parsed.fieldCorrection;
   if ( typeof parsed.drizzleEnabled == "boolean" )
      config.drizzleEnabled = parsed.drizzleEnabled;
   if ( Array.isArray( parsed.excludeFrames ) )
      config.excludeFrames = parsed.excludeFrames.filter( path => typeof path == "string" && path.trim().length > 0 );
   if ( [ "max", "band", "ctn" ].indexOf( parsed.rFieldMode ) >= 0 )
      config.rFieldMode = parsed.rFieldMode;
   for ( let field of dwtNumericSettings() )
   {
      let value = parsed[ field.key ];
      if ( Number.isFinite( value ) && field.valid( value ) )
         config[ field.key ] = value;
   }
}

function dwtNumericSettings()
{
   return [
      { page: "Stack", key: "clarityExponent", label: "Clarity A", initial: 20, valid: v => v >= 0,
         hint: "Enter 0 or greater." },
      { page: "Stack", key: "rejectionSigma", label: "Rejection k", initial: 3.5, valid: v => v >= 0,
         hint: "Enter 0 or greater. Zero disables pixel rejection." },
      { page: "Stack", key: "registrationStars", label: "Catalog stars", initial: 240,
         valid: v => Number.isInteger( v ) && v >= 8 && v <= 4096, hint: "Enter 8 to 4096 stars per frame." },
      { page: "Stack", key: "frameLimit", label: "Frame limit", initial: 0,
         valid: v => Number.isSafeInteger( v ) && v >= 0, hint: "Use the first N active frames per WBPP group. Zero uses all." },
      { page: "Stack", key: "drizzleScale", label: "Drizzle scale", initial: 1, valid: v => v > 0,
         hint: "Enter a value greater than 0." },
      { page: "Stack", key: "drizzlePixfrac", label: "Pixfrac", initial: 1, valid: v => v > 0 && v <= 1,
         hint: "Enter a value greater than 0 and at most 1." },
      { page: "Stack", key: "fwhmMax", label: "FWHM limit (px)", initial: 0, valid: v => v >= 0,
         hint: "Reject frames above this FWHM. Zero disables the limit." },
      { page: "Stack", key: "referenceFrames", label: "Reference frames", initial: 0,
         valid: v => Number.isInteger( v ) && v >= 0 && v <= 64, hint: "Enter 0 to 64. Zero uses up to eight sharp frames." },
      { page: "PWS", key: "snrExponent", label: "SNR exponent B", initial: 2, valid: v => v >= 0,
         hint: "Enter 0 or greater." },
      { page: "PWS", key: "snrClip", label: "SNR weight clip", initial: 3, valid: v => v >= 0,
         hint: "Values at or below 1 disable clipping." },
      { page: "PWS", key: "detailSnr", label: "Detail SNR β", initial: 0, valid: v => true,
         hint: "Enter 1 for the Python GUI's bright-sky compensation." },
      { page: "PWS", key: "fluxPower", label: "Flux exponent", initial: 0, valid: v => true,
         hint: "Zero disables flux weighting." },
      { page: "PWS", key: "sharpWeight", label: "Global sharpness", initial: 0, valid: v => v >= 0,
         hint: "Enter an additional global clarity exponent. Zero disables it." },
      { page: "PWS", key: "rejectionFraction", label: "Max rejected fraction", initial: 1.0 / 3.0,
         valid: v => v >= 0 && v <= 1, hint: "Enter a fraction from 0 to 1." },
      { page: "PWS", key: "stackWorkers", label: "Stacking workers", initial: 0,
         valid: v => Number.isInteger( v ) && v >= 0 && v <= 8, hint: "Enter 0 to 8. Zero selects up to eight workers automatically." },
      { page: "R field", key: "rFieldBlur", label: "Blur σ (px)", initial: 0, valid: v => v >= 0 && v <= 20,
         hint: "Enter 0 to 20. A positive value uses a smoothed local-peak field." },
      { page: "R field", key: "rFieldFloor", label: "R floor", initial: 0, valid: v => v >= 0 && v <= 0.95,
         hint: "Enter a fraction from 0 to 0.95." },
      { page: "R field", key: "rFieldWindow", label: "Band-pass window", initial: 0, valid: v => v >= 0,
         hint: "Enter 0 or greater. Zero selects an automatic window." },
      { page: "R field", key: "rFieldContrast", label: "Contrast threshold", initial: 3, valid: v => v > 0.8,
         hint: "Enter a value greater than 0.8 for contrast-to-noise mode." },
      { page: "R field", key: "envWindowMultiplier", label: "Envelope window ×", initial: 2, valid: v => v > 0,
         hint: "Enter a positive window multiplier." },
      { page: "R field", key: "envEpsilon", label: "Envelope noise ε", initial: 16, valid: v => v >= 0,
         hint: "Enter 0 or greater." },
      { page: "R field", key: "envPower", label: "Envelope exponent", initial: 1, valid: v => v > 0,
         hint: "Enter a positive exponent." }
   ];
}

function sanitizeDwtName( value )
{
   return String( value ).replace( /[^A-Za-z0-9_.-]+/g, "_" );
}

function dwtPathKey( path )
{
   return String( path ).trim().replace( /\\/g, "/" ).toLowerCase();
}

function dwtSourceForFrame( frame )
{
   if ( !frame || !frame.fileItem )
      return undefined;
   let processed = frame.fileItem.processed && frame.fileItem.processed[ frame.associatedID ];
   if ( processed )
   {
      if ( processed[ BPP.FrameProcessingStep.DEBAYER ] )
         return processed[ BPP.FrameProcessingStep.DEBAYER ];
      if ( processed[ BPP.FrameProcessingStep.CALIBRATION ] )
         return processed[ BPP.FrameProcessingStep.CALIBRATION ];
   }
   let defaults = frame.fileItem.processed && frame.fileItem.processed[ BPP.Defaults.associatedId ];
   if ( defaults )
   {
      if ( defaults[ BPP.FrameProcessingStep.DEBAYER ] )
         return defaults[ BPP.FrameProcessingStep.DEBAYER ];
      if ( defaults[ BPP.FrameProcessingStep.CALIBRATION ] )
         return defaults[ BPP.FrameProcessingStep.CALIBRATION ];
   }
   return undefined;
}

function findDwtInsertionIndex()
{
   let operations = engine.operationQueue.operations;
   let lastRegistration = -1;
   let firstIntegration = operations.length;
   for ( let i = 0; i < operations.length; ++i )
   {
      let operation = operations[ i ] && operations[ i ].operation;
      if ( !operation || !operation.envForScript )
         continue;
      let info = operation.envForScript();
      if ( !info || !info.group || info.group.imageType != ImageType.Light )
         continue;
      let name = String( info.name || "" );
      if ( name == "Registration" )
         lastRegistration = i;
      if ( name == "Integration" || name == "FastIntegration"
         || name == "Drizzle Integration" )
         firstIntegration = Math.min( firstIntegration, i );
   }
   if ( lastRegistration >= 0 )
      return Math.min( lastRegistration + 1, firstIntegration );
   return firstIntegration;
}

function dwtInterruptionRequested()
{
   CoreApplication.processEvents();
   return engine.operationQueue.interruptionRequested;
}

function runExternalProcess( executable, args, onProgress )
{
   let process = new ExternalProcess;
   let stdout = "";
   let stderr = "";
   let started = false;
   let exitCode = null;
   let lastProgress = 0;

   function lineReader( append )
   {
      let pending = "";
      return function( chunk, finished )
      {
         let lines = ( pending + chunk ).split( "\n" );
         pending = lines.pop();
         if ( finished && pending.length > 0 )
         {
            lines.push( pending );
            pending = "";
         }
         for ( let line of lines )
         {
            line = line.replace( /\r$/, "" );
            let progress = /^\[DWT\] (\d+) (.*)$/.exec( line );
            if ( progress != null )
            {
               let sequence = Number( progress[ 1 ] );
               if ( sequence > lastProgress )
               {
                  lastProgress = sequence;
                  onProgress( progress[ 2 ] );
               }
            }
            else
               append( line + "\n" );
         }
      };
   }

   let readStdout = lineReader( line => { stdout += line; } );
   let readStderr = lineReader( line => { stderr += line; } );
   process.onStarted = () => { started = true; };
   process.onFinished = ( code ) => { exitCode = code; };
   process.onStandardOutputDataAvailable = () => { readStdout( String( process.stdout ), false ); };
   process.onStandardErrorDataAvailable = () => { readStderr( String( process.stderr ), false ); };

   try
   {
      if ( dwtInterruptionRequested() )
         return { canceled: true };
      if ( process.start( executable, args ) === false )
         return { ok: false, stdout: stdout, stderr: stderr };
      while ( process.isStarting || process.isRunning )
         if ( dwtInterruptionRequested() )
            return { canceled: true };
      if ( engine.operationQueue.interruptionRequested )
         return { canceled: true };
      readStdout( String( process.stdout ), true );
      readStderr( String( process.stderr ), true );
      return { ok: started && exitCode === 0, stdout: stdout, stderr: stderr };
   }
   catch ( error )
   {
      if ( engine.operationQueue.interruptionRequested )
         return { canceled: true };
      return { ok: false, stdout: stdout, stderr: stderr + error.message };
   }
   finally
   {
      if ( process.isStarting )
         process.waitForStarted( 250 );
      if ( process.isRunning )
      {
         process.terminate();
         process.waitForFinished( 250 );
      }
      if ( process.isStarting || process.isRunning )
      {
         process.kill();
         process.waitForFinished( 1000 );
      }
   }
}

function writeCalibratedFloat32( sourcePath, outputPath, width, height )
{
   let window = ImageWindow.open( sourcePath )[ 0 ];
   if ( window == null || window.isNull )
      return false;
   let image = window.mainView.image;
   if ( image.numberOfNominalChannels != 1 || image.width != width || image.height != height )
   {
      window.forceClose();
      return false;
   }
   image.resetSelections();
   image.selectedChannel = 0;
   let samples = new Float32Array( image.width * image.height );
   image.getSamples( samples );
   window.forceClose();
   let file = File.createFileForWriting( outputPath );
   file.write( samples );
   file.close();
   return true;
}

function readFloat32Window( sourcePath, width, height )
{
   if ( !File.exists( sourcePath ) )
      return null;
   let file = File.openFileForReading( sourcePath );
   let samples = file.read( DataType.Float32Array, width * height );
   file.close();
   let window = new ImageWindow( width, height, 1, 32, true, false, "DWT_PWS" );
   window.mainView.beginProcess( UndoFlag.NoSwapFile );
   window.mainView.image.setSamples( samples );
   window.mainView.endProcess();
   window.show();
   return window;
}
