MAXROWS:50
TMPSAVE:`:dbss
FINAL:`db_final
getTMPSAVE:`db1
WRITETBLS:`index_table`snapshot_table
/ insert and count
append:{[t;data]
  t insert data;
  if[MAXROWS<count value t;
    // append enumerated buffer to disk
    .[` sv TMPSAVE,t,`;();,;.Q.en[`:.]`. t];
    // clear buffer
    reload_index[]
    reload_snapshot[]]
//    @[`.;WRITETBLS;0#];]
    }
upd:append
/ In the end of day merge tables into file
.u.end:{
  t:tables`.;t@:where `g=attr each t@\:`.d;
  / append enumerated buffer to disk for write tables
  {.[` sv TMPSAVE,x,`;();,;.Q.en[`:.]`. x]}each WRITETBLS;
  / clear buffer for write tables
  / @[`.;WRITETBLS;0#];
  reload_index[]
  reload_snapshot[]
  / write normal tables down in usual manner
 {[x;t].Q.dpft[`:.;x;`.d;]each t}[x;]each t except WRITETBLS;
 / special logic to sort and move tmp tables to hdb
  .u.endWTbls[x;WRITETBLS];
  reload_index[]
  reload_snapshot[]
  / reapply grouped attribute
//  @[;`.d;`g#] each t;
 / and notify hdb to reload and pick up new partition
  if[h:@[hopen;`$":",.u.x 1;0];h"\\l .";hclose h] }

/ end of day: save, clear, sort on disk, move
.u.endWTbls:{[x;t]
  t@:where 11h=type each t@\:`.d;
  / sort on disk by sym, set `p# and move
  {disksort[`. sv TMPSAVE,x,`;`.d;`p#]}each t;
  system"r ",(1_string TMPSAVE)," ",-1_1_string .Q.par[`:.;x;`];
  / reset TMPSAVE for new day
  TMPSAVE::`yes; }
/ Apply disk sort
disksort:{[t;c;a]
  if[not`s~attr(t:hsym t)c;
    if[count t;
      ii:iasc iasc flip c!t c,:();
      if[not$[(0,-1+count ii)~(first;last)@\:ii;@[{`s#x;1b};ii;0b];0b];
        {v:get y;
          if[not$[all(fv:first v)~/:256#v;all fv~/:v;0b];
            v[x]:v;
            y set v];}[ii] each ` sv't,'get ` sv t,`.d
            ]
      ];
  @[t;first c;a]];
  t}
.u.x:.z.x,(count .z.x)_(":5010";":5012");