MAXROWS:50
TMPSAVE:`asd :: .z.d
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
    @[`.;t;0#] ] }
upd:append
/ In the end of day merge tables into file
.u.end:{
  t:tables`.;
  t@:where 11h=type each t@\:`sym;
  / append enumerated buffer to disk
  {.[` sv TMPSAVE,x,`;();,;.Q.en[`:.]`. x]}each t; / clear buffer
  /  @[`.;t;0#];
  reload_index[]
  reload_snapshot[]
  / sort on disk by sym and set `p#
  / {@[`sym xasc` sv TMPSAVE,x,`;`sym;`p#]}each t;
  {disksort[` sv TMPSAVE,x,`;`sym;`p#]}each t;
  / move the complete partition to final home,
  / use mv instead of built-in r if filesystem whines
  system"r ",(1_string TMPSAVE)," ",-1_1_string .Q.par[`:.;x;`];
  / reset TMPSAVE for new day
  TMPSAVE::getTMPSAVE .z.d;
  / and notify hdb to reload and pick up new partition
  if[h:@[hopen;`$":",.u.x 1;0];h"\\l .";hclose h]; }

/ end of day: save, clear, sort on disk, move
.u.endWTbls:{[x;t]
  t@:where 11h=type each t@\:`.d;
  / sort on disk by sym, set `p# and move
  {disksort[` sv TMPSAVE,x,`;`.d;`p#]}each t;
  system"r ",(1_string FINAL)," ",-1_1_string .Q.par[`:.;x;`];
  / reset TMPSAVE for new day
  FINAL::TMPSAVE .z.d; }
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