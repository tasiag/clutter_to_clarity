params = SetQuestParamsPipe(ndims(Data), false);
[ Trees, dual_aff, init_aff, embedding ] = RunGenericDimsQuestionnaire( params, Data );
emb = embedding{1};

figure
scatter3(emb(:,1),emb(:,2),emb(:,3))
plotEmbeddingWithColors(emb, ex, "experiments")
shg

figure
emb = embedding{2};
scatter3(emb(:,1),emb(:,2),emb(:,3))
plotEmbeddingWithColors(emb, z, "z")
shg

figure
emb = embedding{3};
scatter3(emb(:,1),emb(:,2),emb(:,3))
plotEmbeddingWithColors(emb, t, "time")


