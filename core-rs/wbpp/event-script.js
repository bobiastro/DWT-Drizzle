// Optional WBPP event script for DWT-Drizzle diagnostics.
//
// This file only observes the official WBPP pipeline. It does not run PWS or
// drizzle and does not alter WBPP frame selection.

function onPipelineStart( env, state )
{
   state.started = new Date();
   state.dwt = [];
   console.noteln( "* DWT-Drizzle observer: pipeline started." );
}

function onOperationStart( env, state )
{
   if ( env.name == "Registration" || env.name == "Drizzle Integration" )
   {
      state.dwt.push( {
         name: env.name,
         index: env.operationIndex,
         started: new Date()
      } );
      console.noteln( "* DWT-Drizzle observer: ", env.name );
   }
}

function onOperationDone( env, state )
{
   if ( env.name == "Registration" || env.name == "Drizzle Integration" )
   {
      let status = env.status == OperationBlockStatus.DONE ? "done" : "not done";
      console.noteln( "* DWT-Drizzle observer: ", env.name, " ", status );
   }
}

function onPipelineEnd( env, state )
{
   console.noteln( "* DWT-Drizzle observer: pipeline ended." );
}
