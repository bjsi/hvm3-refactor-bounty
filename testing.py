from main import FileContext

code = """
collapseDupsAt :: IM.IntMap [Int] -> ReduceAt -> Book -> Loc -> HVM Core
-- BLOCK 12:
collapseDupsAt state@(paths) reduceAt book host = unsafeInterleaveIO $ do
  term <- reduceAt book host
  case tagT (termTag term) of
-- BLOCK 13:
    ERA -> do
      return Era
-- BLOCK 14:
    LET -> do
      let loc = termLoc term
      let mode = modeT (termLab term)
      name <- return $ "$" ++ show (loc + 0)
      val0 <- collapseDupsAt state reduceAt book (loc + 1)
      bod0 <- collapseDupsAt state reduceAt book (loc + 2)
      return $ Let mode name val0 bod0
-- BLOCK 15:
    LAM -> do
      let loc = termLoc term
      name <- return $ "$" ++ show (loc + 0)
      bod0 <- collapseDupsAt state reduceAt book (loc + 0)
      return $ Lam name bod0
-- BLOCK 16:
    APP -> do
      let loc = termLoc term
      fun0 <- collapseDupsAt state reduceAt book (loc + 0)
      arg0 <- collapseDupsAt state reduceAt book (loc + 1)
      return $ App fun0 arg0
-- BLOCK 17:
    SUP -> do
      let loc = termLoc term
      let lab = termLab term
      case IM.lookup (fromIntegral lab) paths of
        Just (p:ps) -> do
          let newPaths = IM.insert (fromIntegral lab) ps paths
          collapseDupsAt (newPaths) reduceAt book (loc + fromIntegral p)
        _ -> do
          tm00 <- collapseDupsAt state reduceAt book (loc + 0)
          tm11 <- collapseDupsAt state reduceAt book (loc + 1)
          return $ Sup lab tm00 tm11
-- BLOCK 18:
    VAR -> do
      let loc = termLoc term
      sub <- got loc
      if termGetBit sub /= 0
      then do
        set (loc + 0) (termRemBit sub)
        collapseDupsAt state reduceAt book (loc + 0)
      else do
        name <- return $ "$" ++ show loc
        return $ Var name
-- BLOCK 19:
    DP0 -> do
      let loc = termLoc term
      let lab = termLab term
      sb0 <- got (loc+0)
      if termGetBit sb0 /= 0
      then do
        set (loc + 0) (termRemBit sb0)
        collapseDupsAt state reduceAt book (loc + 0)
      else do
        let newPaths = IM.alter (Just . maybe [0] (0:)) (fromIntegral lab) paths
        collapseDupsAt (newPaths) reduceAt book (loc + 0)
-- BLOCK 20:
    DP1 -> do
      let loc = termLoc term
      let lab = termLab term
      sb1 <- got (loc+1)
      if termGetBit sb1 /= 0
      then do
        set (loc + 1) (termRemBit sb1)
        collapseDupsAt state reduceAt book (loc + 1)
      else do
        let newPaths = IM.alter (Just . maybe [1] (1:)) (fromIntegral lab) paths
        collapseDupsAt (newPaths) reduceAt book (loc + 0)
-- BLOCK 21:
    CTR -> do
      let loc = termLoc term
      let lab = termLab term
      let cid = u12v2X lab
      let ari = u12v2Y lab
      let aux = if ari == 0 then [] else [loc + i | i <- [0..ari-1]]
      fds0 <- forM aux (collapseDupsAt state reduceAt book)
      return $ Ctr cid fds0
-- BLOCK 22:
    MAT -> do
      let loc = termLoc term
      let len = u12v2X $ termLab term
      let aux = if len == 0 then [] else [loc + 1 + i | i <- [0..len-1]]
      val0 <- collapseDupsAt state reduceAt book (loc + 0)
      css0 <- forM aux $ \\h -> do
        bod <- collapseDupsAt state reduceAt book h
        return $ ("#", [], bod) -- TODO: recover constructor and fields
      return $ Mat val0 [] css0
-- BLOCK 23:
    W32 -> do
      let val = termLoc term
      return $ U32 (fromIntegral val)
-- BLOCK 24:
    CHR -> do
      let val = termLoc term
      return $ Chr (chr (fromIntegral val))
-- BLOCK 25:
    OPX -> do
      let loc = termLoc term
      let opr = toEnum (fromIntegral (termLab term))
      nm00 <- collapseDupsAt state reduceAt book (loc + 0)
      nm10 <- collapseDupsAt state reduceAt book (loc + 1)
      return $ Op2 opr nm00 nm10
-- BLOCK 26:
    OPY -> do
      let loc = termLoc term
      let opr = toEnum (fromIntegral (termLab term))
      nm00 <- collapseDupsAt state reduceAt book (loc + 0)
      nm10 <- collapseDupsAt state reduceAt book (loc + 1)
      return $ Op2 opr nm00 nm10
-- BLOCK 27:
    REF -> do
      let loc = termLoc term
      let lab = termLab term
      let fid = u12v2X lab
      let ari = u12v2Y lab
      arg0 <- mapM (collapseDupsAt state reduceAt book) [loc + i | i <- [0..ari-1]]
      let name = MS.findWithDefault "?" fid (idToName book)
      return $ Ref name fid arg0
-- BLOCK 28:
    tag -> do
      putStrLn ("unexpected-tag:" ++ show tag)
      return $ Var "?"
      -- exitFailure
-- BLOCK 29:
-- Sup Collapser
-- -------------
hello :: Int -> Int
hello x = x + 1
-- -------------
data Dag a =
  | Leaf a
  | Node (Tag a) (Tag a)
  | Ctr (Int, [Tag a])
  | Mat (Tag a) [(String, [Tag a])]
"""

ctx = FileContext(file=".hs", content=code)
print(ctx.stringify())
ctx = ctx.show_scopes_mentioning("Book", scope="full", parents="all")
print(ctx.stringify())