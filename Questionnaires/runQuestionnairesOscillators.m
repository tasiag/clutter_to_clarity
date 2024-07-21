params = SetQuestParamsOscillators(ndims(Data), false);
[ Trees, dual_aff, init_aff, embedding ] = RunGenericDimsQuestionnaire( params, Data );

figure
emb = embedding{1};
scatter3(emb(:,1),emb(:,2),emb(:,3))
plotEmbeddingWithColors(emb, K, "Coupling Strength")
shg

figure
emb = embedding{2};
scatter3(emb(:,1),emb(:,2),emb(:,3))
plotEmbeddingWithColors(emb, x, "space (omega)")
shg

figure
emb = embedding{3};
scatter3(emb(:,1),emb(:,2),emb(:,3))
plotEmbeddingWithColors(emb, t, "time")


